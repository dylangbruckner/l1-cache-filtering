#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>

#include "pin.H"


// Number of bits in the address
const static UINT64 ADDRESS_SIZE = 64;

// Defines the Structure of a Cache Entry
// LRU_status stores the adges of a cache line
// if LRU_status == 0, the cache line is Most Recently Used.
// else if LRU_status == assoc - 1, the cache line is one of the oldest cache lines.
struct cacheEntry
{
    int LRU_status;
    unsigned long Tag;
    bool Valid;
};

struct PrefetchEntry {
    bool valid;
    unsigned long pc_tag;       // which PC this entry belongs to
    unsigned long last_line;    // last line address (addr >> blockOffsetSize)
    long stride;                // stride in bytes
    uint8_t confidence;         // 0..3
    int lru;                    // 0 = MRU, higher = older (for assoc > 1)
};

class prefetch_table {
public:
    prefetch_table(int num_entries, int associativity, int blockSize);

    virtual ~prefetch_table() {};

    // updates the prefetch table with pc and addr access
    void update(unsigned long pc, unsigned long addr);

    // returns true if you should prefetch for the specified pc on a miss
    bool should_prefetch(unsigned long pc) const;

    // the stride for a given entry in the pc table
    long get_stride(unsigned long pc) const;

private:
    int entries;       // total entries = sets * assoc
    int assoc;         // ways per set
    int sets;          // number of sets = entries / assoc
    int blockSize;     // cache block size in bytes
    int blockOffsetBits;

    PrefetchEntry* table;

    int pc_index(unsigned long pc) const;            // returns set index
    PrefetchEntry* find_entry(unsigned long pc, int set) const;
    PrefetchEntry* find_victim(int set);
    void update_lru(int set, int way);
};

class insertion_policy {
public:
    virtual ~insertion_policy() {}

    // Called on every access (hit or miss) at this cache level.
    virtual void on_access(unsigned long pc, unsigned long addr, bool hit) {}

    // Called when we are about to insert a new block on a miss.
    // Returns true if the block should go into this cache, false if it should be filtered out.
    virtual bool should_insert(unsigned long pc, unsigned long addr) = 0;
};

class bernoulli_insertion_policy : public insertion_policy {
public:
    bernoulli_insertion_policy(double p, uint64_t seed = 1)
        : dist(p), gen(seed) {}

    bool should_insert(unsigned long pc, unsigned long addr) override {
        (void)pc; (void)addr;
        return dist(gen);  // true with probability p
    }

private:
    std::bernoulli_distribution dist;
    std::mt19937_64 gen;
};

// forward declarations for different extras that caches can contain
class victim_cache;
class prefetch_table;
class prefetch_buffer;

class cache
{
public:
    virtual ~cache() {};

    // Access the generic cache.
    // You might want to override this in L1D to access the victim cache.
    virtual void addressRequest( unsigned long address, unsigned long pc);

    // attempts to find the target_address in the victim cache and swap it for the victim_address. returns false if 
    // the target address is not in the victim cache, or if the cache is not a victim cache
    virtual bool swapElements( unsigned long victim_address, unsigned long target_address, bool has_evicted, unsigned long pc);

    // checks if the given address exists in the prefetcher and returns true if it does, invalidating it
    virtual bool prefAddressRequest(unsigned long address);

    virtual bool set_is_full(unsigned int setIndex);

    // Get Statistics Methods
    virtual UINT64 getTotalMiss();
    virtual UINT64 getHit();
    virtual UINT64 getRequest();
    virtual UINT64 getEntryRemoved();

    // Get Settings Methods
    virtual int getCacheSize();
    virtual int getCacheAssoc();
    virtual int getCacheBlockSize();
    virtual unsigned int getTagSize() {return tagSize;}
    virtual unsigned int getBlockOffsetSize() {return blockOffsetSize;}
    virtual unsigned int getSetSize() {return setSize;}

    void set_insertion_policy(insertion_policy* p) { policy = p; }
    void set_insertion_depth(int d) { insertion_depth = d; }


protected:
    cache( int blockSize, int totalCacheSize, int associativity, cache* nextLevel, bool writebackDirty, 
      victim_cache* victim = nullptr,
      prefetch_table* pref = nullptr,
      prefetch_buffer* pbuf = nullptr,
      int prefetch_count = 0); 

    //Calculate the Tag and Set of an address based on this cache's properties
    unsigned int getTag( unsigned int address );
    unsigned int getSet( unsigned int address );

    // returns -1 for a miss
    // index into array for a hit
    int isHit( unsigned int tagBits, unsigned int setBits );

    // Set a certain index as the Most Receintly Used in its
    // associativity, and adjusts all the appropriate indices to match
    // the LRU scheme
    void updateLRU( int setBits, int MRU_index );

    // Gets the index of the LRU index for a given set of setBits
    int getLRU( int setBits );

    // Gets the index of the MRU index for a given set of setBits
    int getMRU( int setBits );
    void updateLRU_with_depth(int setBits, int index, int depth_from_mru);

    // Initializes this cache
    void clearCache();

    void addTotalMiss();
    void addHit();
    void addRequest();
    void addEntryRemoved();

    // additional cache workings
    bool hasVictimCache() const { return victim != nullptr; }
    bool hasPrefetcher() const { return prefetcher != nullptr && pbuffer != nullptr; }
    bool checkVictim(unsigned long address);
    bool checkPrefetchBuffer(unsigned long address);
    void updatePrefetcher(unsigned long pc, unsigned long address);
    void issuePrefetches(unsigned long pc, unsigned long address);

    // Given Properties
    const int blockSz;
    const int totalCacheSz;
    const int assoc;

    // Bit Field Sizes
    const unsigned int blockOffsetSize;
    const unsigned int setSize;
    const unsigned int tagSize;

    const unsigned int tagMask;
    const unsigned int setMask;
    const int maxSetValue;

    // Statistics
    UINT64 totalMisses;
    UINT64 hits;
    UINT64 requests;
    UINT64 entriesKickedOut;

    // The actual cache array
    cacheEntry* cacheMem;

    // The next level in the cache hierachy
    cache* const nextLevel;
    // Does this cache write evicted items to the next level (icaches don't need to)
    const bool writebackDirty;

    // different optional additional caches
    victim_cache* victim;
    prefetch_table* prefetcher;
    prefetch_buffer* pbuffer;
    const int prefetch_count;

    insertion_policy* policy;
    int insertion_depth;
};

class memory : public cache {
public:
    memory() :
        cache(1, 1, 1, nullptr, false)
    { }

    void addressRequest( unsigned long address, unsigned long pc) {
        (void) address;
        (void) pc;
        addRequest();
    }
};

class victim_cache : public cache {
public:
    victim_cache(int blockSize, int totalCacheSize, cache *nextLevel, bool writebackDirty) :
        cache( blockSize, totalCacheSize, totalCacheSize / blockSize, nextLevel, writebackDirty)
    { }

    void addressRequest( unsigned long address, unsigned long pc) {
        (void) address;
        return;
    }

    bool swapElements( unsigned long victim_address, unsigned long target_address, bool has_evicted, unsigned long pc) override;
    bool containsElement(unsigned long target_address);
};

class prefetch_buffer : public cache {
    public:
    prefetch_buffer( int blockSize, int totalCacheSize) :
        cache( blockSize, totalCacheSize, totalCacheSize / blockSize, nullptr, false)
    { }
    void addressRequest (unsigned long address, unsigned long pc) override;
    bool prefAddressRequest(unsigned long address) override;
};
class filter_cache : public cache {
public:
    filter_cache(int blockSize, int totalCacheSize, int associativity, cache* nextLevel, bool writebackDirty)
        : cache(blockSize, totalCacheSize, associativity,
                nextLevel, writebackDirty)
    { }
};

class l1icache : public cache {
public:
    l1icache( int blockSize, int totalCacheSize, int associativity, cache *nextLevel) :
        cache( blockSize, totalCacheSize, associativity, nextLevel, false)
    { }
};

class l1dcache : public cache {
public:
    l1dcache( int blockSize, int totalCacheSize, int associativity, cache *nextLevel, victim_cache *victim, prefetch_table *table, prefetch_buffer *buffer, int buf_size) :
        cache( blockSize, totalCacheSize, associativity, nextLevel, true, victim, table, buffer, buf_size)
    { }
};

class l2cache : public cache {
public:
    l2cache(int blockSize, int totalCacheSize, int associativity, cache *nextLevel) :
        cache( blockSize, totalCacheSize, associativity, nextLevel, true)
    { }
};


using namespace std;
/*
This code implements a simple cache simulator. In this implementation, we
assume there is L1 Instruction Cache, L1 Data Cache, and L2 Cache.

We have a base class declared in cache.h file and three derived classes:
L1InstCache, L1DataCache, and L2Cache.

We have a simple memory class to track the number of memory requests.

This is a simplified cache model since we do not differentiate read and write
accesses, which is not true for real cache system.

We only track the number of hits and misses in both L1 and L2 caches.

In both L1 and L2 caches, we use a simple LRU algorithm as cache replacement policy.

*/

/* ===================================================================== */
/* Commandline Switches */
/* ===================================================================== */
KNOB<string> KnobOutputFile(KNOB_MODE_WRITEONCE,    "pintool",
        "outfile", "cache.out", "Cache results file name");

KNOB<string> KnobConfigFile(KNOB_MODE_WRITEONCE,    "pintool",
        "config", "config-base", "Configuration file name");

KNOB<UINT64> KnobInstructionCount(KNOB_MODE_WRITEONCE, "pintool",
        "max_inst","1000000000", "Number of instructions to profile");

// Globals for the various caches
/* ===================================================================== */
l1icache* icache;
l1dcache* dcache;
l2cache* llcache;
memory* mem;

// Keep track if instruction counts so we know when to end simmulation
UINT64 icount;

void PrintResults(void);



////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
    


////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
cache::cache( int blockSize, int totalCacheSize, int associativity, cache* nextLevel, bool writebackDirty, 
      victim_cache* victim,
      prefetch_table* pref,
      prefetch_buffer* pbuf,
      int prefetch_count) :
    // Set Cache properties
    blockSz(blockSize),
    totalCacheSz(totalCacheSize),
    assoc(associativity),
    
    // Calculate Cache bit sizes and masks
    blockOffsetSize(log2(blockSize)),
    setSize(log2(totalCacheSize / (blockSize * associativity))),
    tagSize(ADDRESS_SIZE - blockOffsetSize - setSize),
    tagMask( (1 << tagSize) - 1),
    setMask( (1 << setSize) - 1),
    maxSetValue((int) 1 << setSize),
    // Next level properties
    nextLevel(nextLevel),
    writebackDirty(writebackDirty),

    // set helper caches
    victim(victim),
    prefetcher(pref),
    pbuffer(pbuf),
    prefetch_count(prefetch_count),

    policy(nullptr), 
    insertion_depth(0)
{
    // Allocate memory for the cache array
    cacheMem = new cacheEntry[totalCacheSize/blockSize];

    clearCache();

    // Clear the statistics
    totalMisses = 0;
    hits = 0;
    requests = 0;
    entriesKickedOut = 0;
}

void cache::clearCache()
{
    // Loop through entire cache array
    for( int i = 0; i < (maxSetValue) * assoc; i++ ) {
        cacheMem[ i ].LRU_status = (i % assoc);
        cacheMem[ i ].Tag = 0;
        cacheMem[ i ].Valid = false;
    }
}

unsigned int cache::getTag( unsigned int address )
{
    unsigned int ret = (address >> (blockOffsetSize + setSize)) & tagMask;
    return ret;
}

unsigned int cache::getSet( unsigned int address )
{
    // Bit Mask to get setBits
    unsigned int ret = (address >> (blockOffsetSize)) & setMask;
    return ret;
}

int cache::isHit( unsigned int tagBits, unsigned int setIndex)
{
    ///cout << "isHit.b" << endl;
    int result = -1;

    // Loop Through By Associativity
    for( int i = 0; i < assoc; i++ )
    {
        // Check if the cache location contains the requested data
        if( cacheMem[ (i + setIndex * assoc) ].Valid == true &&
                cacheMem[ (i + setIndex * assoc) ].Tag == tagBits )
        {
            return i;
            break;
        }
    }

    return result;
}

//
// Update the LRU for the system
// Input:
//  setBits - The set field of the current address
//  MRU_index - The index into the cache's array of the Most Receintly
//     Used Entry (which should be i * setBits for some int i).
// Results:
//  The entry and MRU_index will be 0 to show that it is the MRU.
//  All other entries will be updated to reflect the new MRU.
//
void cache::updateLRU( int setBits, int MRU_index )
{
    int upperBounds = assoc - 1;

    // Update all of the other places necesary to accomidate the change
    for( int i = 0; i < assoc; i++ )
    {
        if( cacheMem[ i + setBits*assoc ].LRU_status >= 0 &&
                cacheMem[ i + setBits*assoc ].LRU_status < upperBounds )
        {
            cacheMem[ i + setBits*assoc ].LRU_status++;
        }
    }

    // Set the new MRU location to show that it is the MRU
    cacheMem[ MRU_index + setBits*assoc ].LRU_status = 0;
}

void cache::updateLRU_with_depth(int setBits, int index, int depth_from_mru) {
    int max_pos = assoc - 1;
    if (depth_from_mru < 0) depth_from_mru = 0;
    if (depth_from_mru > max_pos) depth_from_mru = max_pos;

    int target_pos = depth_from_mru;
    int base = setBits * assoc;

    // Shift others: any line with LRU_status >= target_pos and < max_pos
    // moves one step older (pos++).
    for (int i = 0; i < assoc; ++i) {
        if (i == index) continue;
        int pos = cacheMem[base + i].LRU_status;
        if (pos >= target_pos && pos < max_pos) {
            cacheMem[base + i].LRU_status = pos + 1;
        }
    }

    cacheMem[base + index].LRU_status = target_pos;
}
//
// Input:
//   setBits - The set field of the address
// Output:
//   (int) - The index into the cache of the Least Recently Used
//     value for the given setBits field.
//    -1 If there is an error
//
int cache::getLRU( int setBits )
{
    for( int i = 0; i < assoc; i++ )
    {
        if( cacheMem[ i + setBits*assoc ].LRU_status == (assoc - 1) )
            return i;
    }
    return -1;
}

//
// Input:
//   setBits - The set field of the address
// Output:
//   (int) - The index into the cache of the Most Recently Used
//     value for the given setBits field.
//    -1 If there is an error
//
int cache::getMRU( int setBits )
{
    for( int i = 0; i < assoc; i++ )
    {
        if( cacheMem[ i + setBits*assoc ].LRU_status == 0 )
            return i;
    }
    return -1;
}
//
// Mark that the cache Missed
//
void cache::addTotalMiss()
{
    totalMisses++;
}

//
// Mark that the cache Hit
//
void cache::addHit()
{
    hits++;
}

//
// Mark that a memory request was made
//
void cache::addRequest()
{
    requests++;
}

//
// Mark that an entry was kicked out
//
void cache::addEntryRemoved()
{
    entriesKickedOut++;
}

//
// Get the total Miss Counter
//
UINT64 cache::getTotalMiss()
{
    return totalMisses;
}

//
// Get the Hit Counter
//
UINT64 cache::getHit()
{
    return hits;
}

//
// Get the requests Counter
//
UINT64 cache::getRequest()
{
    return requests;
}

//
// Get the removed entry counter
//
UINT64 cache::getEntryRemoved()
{
    return entriesKickedOut;
}

//
// Get the size of the size of the cache
//
int cache::getCacheSize()
{
    return totalCacheSz;
}

//
// Get the associativity of the cache
//
int cache::getCacheAssoc()
{
    return assoc;
}

//
// Get the block size of the cache
//
int cache::getCacheBlockSize()
{
    return blockSz;
}

//
// Access the cache. Checks for hit/miss and updates appropriate stats.
// On a miss, brings the item in from the next level. If necessary,
// writes the evicted item back to the next level.
// Doesn't distinguish between reads and writes.
//
void cache::addressRequest( unsigned long address, unsigned long pc ) {

    // Compute Set / Tag
    unsigned long tagField = getTag( address );
    unsigned long setField = getSet( address );

    // Hit or Miss ?
    int index = isHit( tagField, setField );

    // Count that access
    addRequest();

    if (hasPrefetcher()) prefetcher->update(pc, address); 
    if (policy) policy->on_access(pc, address, index != -1);

    // Miss
    if( index == -1 ) {
        int indexLRU = getLRU( setField );
        unsigned long v_address = 0;
        bool has_evicted = false;

        if(cacheMem[ indexLRU + setField*assoc].Valid) {
            addEntryRemoved();
            unsigned long v_tag = cacheMem[ indexLRU + setField*assoc].Tag;
            v_address = (v_tag << (getSetSize() + getBlockOffsetSize())) |(setField << getBlockOffsetSize());
            has_evicted = true;

        }
        // check victim cache if it exists
        bool found_in_victim = false;
        if (hasVictimCache()) {
            found_in_victim = victim->containsElement(address);
            if (found_in_victim) {
                victim->swapElements(v_address, address, has_evicted, pc);
            }
        }
        bool prefetched = false;
        if (!found_in_victim && hasPrefetcher()) {
            prefetched = pbuffer->prefAddressRequest(address);
            if (!prefetched) {
                long stride = prefetcher->get_stride(pc);
                if (prefetcher->should_prefetch(pc) && stride) {
                    for (int i = 1; i <= prefetch_count; i++) {
                        pbuffer->addressRequest(address + i * stride, pc);
                    }
                }
            }
        }

        // Count that miss
        addTotalMiss();
        // Write the evicted entry to the next level (onlly writeback here if no victim cache, otherwise will be handled there)
        if( writebackDirty &&
            has_evicted && !hasVictimCache()) {
            unsigned long tag = cacheMem[indexLRU + setField*assoc].Tag;
            tag = tag << (getSetSize() + getBlockOffsetSize());
            unsigned long Set = setField;
            Set = Set << (getBlockOffsetSize());
            unsigned long lru_addr = tag + Set;

            // if we are in L1 and have a filter, write back to L2 not the filter
            if (nextLevel->nextLevel && nextLevel->nextLevel->nextLevel) {
                nextLevel->nextLevel->addressRequest(lru_addr, pc);
            }
            else { 
                nextLevel->addressRequest(lru_addr, pc);
            }
        }
        // Load the requested address from next level
        if (!found_in_victim && !prefetched) nextLevel->addressRequest(address, pc);

        // Update LRU / Tag / Valid
        bool insert_here = true;
        if (set_is_full(setField) && policy) {
            insert_here = policy->should_insert(pc, address);
        }

        // if policy says to insert, then insert
        if (!prefetched && (found_in_victim || insert_here)) {
            cacheMem[ indexLRU + setField*assoc].Tag = tagField;
            cacheMem[ indexLRU + setField*assoc].Valid = true;
            if (found_in_victim || insertion_depth <= 0 || !set_is_full(setField)) {
                // Victim promotions, unbiased caches, or not-yet-full sets → true MRU.
                updateLRU(setField, indexLRU);
            } else {
                // Lower-level fill, set already full, and we want biased insertion.
                updateLRU_with_depth(setField, indexLRU, insertion_depth);
            }
            if (hasVictimCache() && !found_in_victim) {
                // add L1 victim to the cache
                victim->swapElements(v_address, address, has_evicted, pc);
            }
        } 
    }
    else {
        // Count that hit
        addHit();

        // Update LRU / Tag / Valid
        updateLRU( setField, index );
    }
}

bool cache::swapElements( unsigned long victim_address, unsigned long target_address, bool has_evicted, unsigned long pc){
    (void) victim_address, (void) target_address, (void) has_evicted, (void) pc;
    return false;
}

bool cache::prefAddressRequest(unsigned long address) {
    (void) address;
    return false;
}

bool cache::set_is_full(unsigned int setIndex) {
    int base = setIndex * assoc;
    for (int i = 0; i < assoc; ++i) {
        if (!cacheMem[base + i].Valid) {
            return false;
        }
    }
    return true;
}

bool victim_cache::containsElement(unsigned long target_address) {
    unsigned long tTagField = getTag(target_address);
    unsigned long setField = 0;

    int hit_index = isHit(tTagField, setField);
    return hit_index != -1;
}
bool victim_cache::swapElements( unsigned long victim_address, unsigned long target_address, bool has_evicted, unsigned long pc){
    unsigned long tTagField = getTag(target_address);
    unsigned long setField = 0;

    int hit_index = isHit(tTagField, setField);
    addRequest();

    if (hit_index != -1) {
        // mark hit
        addHit();
        if (has_evicted) {
            // add L1 victim to the cache
            cacheMem[hit_index].Tag = getTag(victim_address);
            cacheMem[hit_index].Valid = true;
            updateLRU(setField, hit_index);
        } else {
            cacheMem[hit_index].Valid = false;
        }
        return true;
    }
    
    addTotalMiss();

    if (has_evicted) {
        int indexLRU = getLRU(setField);
        if (cacheMem[indexLRU].Valid) {
            addEntryRemoved();
            if (writebackDirty) {
                unsigned long tag = cacheMem[indexLRU].Tag;
                tag = tag << (getSetSize() + getBlockOffsetSize());
                unsigned long Set = setField;
                Set = Set << (getBlockOffsetSize());
                unsigned long lru_addr = tag + Set;
                nextLevel->addressRequest(lru_addr, pc);
            }
        }
        cacheMem[indexLRU].Tag = getTag(victim_address);
        cacheMem[indexLRU].Valid = true;
        updateLRU(setField, indexLRU);
    }

    return false;
}
////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////

prefetch_table::prefetch_table(int num_entries, int associativity, int blkSize)
    : entries(num_entries),
      assoc(associativity),
      sets(num_entries / associativity),
      blockSize(blkSize),
      blockOffsetBits(0)
{
    // compute log2(blockSize)
    int bs = blockSize;
    while ((bs >>= 1) > 0) blockOffsetBits++;

    table = new PrefetchEntry[entries];
    for (int i = 0; i < entries; ++i) {
        table[i].valid = false;
        table[i].pc_tag = 0;
        table[i].last_line = 0;
        table[i].stride = 0;
        table[i].confidence = 0;
        table[i].lru = i % assoc;
    }
}

int prefetch_table::pc_index(unsigned long pc) const {
    // simple hash: lower bits of PC
    return (pc >> 2) & (sets - 1);
}

PrefetchEntry* prefetch_table::find_entry(unsigned long pc, int set) const {
    int base = set * assoc;
    for (int way = 0; way < assoc; ++way) {
        PrefetchEntry* e = &table[base + way];
        if (e->valid && e->pc_tag == pc)
            return e;
    }
    return nullptr;
}

PrefetchEntry* prefetch_table::find_victim(int set) {
    int base = set * assoc;
    int lru_way = 0;
    int max_lru = -1;
    for (int way = 0; way < assoc; ++way) {
        PrefetchEntry* e = &table[base + way];
        if (!e->valid) {
            return e; // free slot
        }
        if (e->lru > max_lru) {
            max_lru = e->lru;
            lru_way = way;
        }
    }
    return &table[base + lru_way];
}

void prefetch_table::update(unsigned long pc, unsigned long addr) {
    unsigned long line = addr >> blockOffsetBits;

    int set = pc_index(pc);
    PrefetchEntry* e = find_entry(pc, set);

    if (!e) {
        // allocate or replace victim
        e = find_victim(set);
        e->valid = true;
        e->pc_tag = pc;
        e->last_line = line;
        e->stride = blockSize;    // default stride = one block
        e->confidence = 0;
        int idx = (int)(e - table);
        int way = idx % assoc;
        update_lru(set, way);
        return;
    }

    // existing entry
    long new_stride_lines = (long)line - (long)e->last_line;
    long new_stride_bytes = new_stride_lines * blockSize;

    bool prediction_correct = false;
    if (e->stride != 0) {
        long predicted_line = (long)e->last_line + (e->stride / blockSize);
        prediction_correct = (predicted_line == (long)line);
    }

    if (prediction_correct) {
        if (e->confidence < 3) {
            e->confidence++;
        }
    } else {
        if (e->confidence == 3) {
            // strong but wrong: just weaken it
            e->confidence--;
        } else {
            // adopt new stride
            e->stride = new_stride_bytes;
            e->confidence = 1;
        }
    }

    e->last_line = line;
    int idx = (int)(e - table);
    int way = idx % assoc;
    update_lru(set, way);
}

void prefetch_table::update_lru(int set, int way) {
    int base = set * assoc;
    int old_lru = table[base + way].lru;

    for (int i = 0; i < assoc; ++i) {
        if (i == way) continue;
        PrefetchEntry& e = table[base + i];
        if (e.valid && e.lru < old_lru) {
            e.lru++;
        }
    }

    table[base + way].lru = 0;
}

bool prefetch_table::should_prefetch(unsigned long pc) const {
    int set = pc_index(pc);
    PrefetchEntry* e = find_entry(pc, set);
    if (!e || !e->valid) return false;
    return e->confidence >= 2;
}

long prefetch_table::get_stride(unsigned long pc) const {
    int set = pc_index(pc);
    PrefetchEntry* e = find_entry(pc, set);
    if (!e || !e->valid) return 0;
    return e->stride;  // bytes, multiple of blockSize
}

bool prefetch_buffer::prefAddressRequest(unsigned long address) {
    unsigned long tTagField = getTag(address);
    unsigned long setField = 0;

    int hit_index = isHit(tTagField, setField);
    addRequest();

    if (hit_index != -1) {
        // mark hit
        addHit();
        cacheMem[hit_index].Valid = false;
        return true;
    }
    addTotalMiss();
    return false;
}

// add the element to the prefetch buffer
void prefetch_buffer::addressRequest(unsigned long address, unsigned long pc) {
    (void) pc;
    unsigned long setField = getSet(address);
    unsigned long tagField = getTag(address);
    int index = isHit(tagField, setField);

    // replace lru item
    if (index == -1) {
        int lru_index = getLRU(setField);
        if(cacheMem[lru_index + setField*assoc].Valid) {
            addEntryRemoved();
        }
        cacheMem[lru_index + setField*assoc].Tag = tagField;
        cacheMem[lru_index + setField*assoc].Valid = true;
        updateLRU( setField, lru_index );
    } else updateLRU(setField, index);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////
/* ===================================================================== */
INT32 Usage()
{
    cerr << "This tool represents a cache simulator.\n"
        "\n";

    cerr << KNOB_BASE::StringKnobSummary();

    cerr << endl;

    return -1;
}


/* ===================================================================== */
victim_cache* vc;
prefetch_buffer* pb;
prefetch_table* pt;
filter_cache* dfcache; 
filter_cache* ifcache;

void CreateCaches(void)
{
    // Check if config file exits
    ifstream config;
    config.open( KnobConfigFile.Value().c_str());
    if(!config.is_open()) {
        cerr << "Cannot open input file : " << KnobConfigFile.Value() << "\n";
        Usage();
        PIN_ExitProcess(EXIT_FAILURE);
    }

    // Create the one and only memory
    mem = new memory();

    // Parse config file and create the three caches
    int i = 0;
    while (!config.eof()){
        string line;
        getline(config, line);
        istringstream parser(line);
        int bsize, csize, assoc, vsize, pbsize, ptsize, ptassoc, pcount, fsize, profilep, fassoc, ins_depth;
        char comma = ',';
        switch(i){
            case 0:
                parser >> bsize >> comma >> csize >> comma >> assoc;
                llcache = new l2cache(bsize, csize, assoc, mem);
                break;
            case 1:
                parser >> bsize >> comma >> csize >> comma >> assoc >> comma >> fsize >> comma >> profilep >> comma >> fassoc;
                
                // add filter in front of L1 
                if (fsize > 0) {
                    ifcache = new filter_cache(bsize, fsize, fassoc, llcache, false);
                    icache = new l1icache(bsize, csize, assoc, ifcache);
                    icache->set_insertion_policy(new bernoulli_insertion_policy(profilep / 100.0));
                } else {
                    ifcache = nullptr;
                    icache = new l1icache(bsize, csize, assoc, llcache);
                }
                
                break;
            case 2:
                parser >> bsize >> comma >> csize >> comma >> assoc >> comma >> vsize >> comma >> pbsize >> comma >> ptsize >> comma >> ptassoc >> comma >> pcount >> comma >> fsize >> comma >> profilep >> comma >> fassoc >> comma >> ins_depth;
                vc = nullptr;
                pb = nullptr;
                pt = nullptr;
                
                if (vsize > 0) {
                    vc = new victim_cache(bsize, vsize, llcache, true);
                } if (pbsize > 0 && ptsize > 0 && ptassoc > 0) {
                    pb = new prefetch_buffer(bsize, pbsize);
                    pt = new prefetch_table(ptsize, ptassoc, bsize);
                }

                // add filter in front of L1
                if (fsize > 0) {
                    dfcache = new filter_cache(bsize, fsize, fassoc, llcache, true);
                    dcache = new l1dcache(bsize, csize, assoc, dfcache, vc, pt, pb, pcount);
                    dcache->set_insertion_policy(new bernoulli_insertion_policy(profilep / 100.0));
                } else {
                    dfcache = nullptr;
                    dcache = new l1dcache(bsize, csize, assoc, llcache, vc, pt, pb, pcount);
                }
                if (ins_depth > 0) dcache->set_insertion_depth(assoc - ins_depth);
                break;
            default:
                break;
        }
        i++;
    }
}

/* ===================================================================== */
void CheckInstructionLimits(void)
{
    if (KnobInstructionCount.Value() > 0 &&
        icount > KnobInstructionCount.Value()) {
        PrintResults();
        PIN_ExitProcess(EXIT_SUCCESS);
    }
}

/* ===================================================================== */
void MemoryOp(ADDRINT pc, ADDRINT address)
{
    dcache->addressRequest( address, pc);
}

/* ===================================================================== */
void AllInstructions(ADDRINT pc, ADDRINT ins_ptr)
{
    (void) pc;
    icount++;
    icache->addressRequest( ins_ptr, 0);

    CheckInstructionLimits();
}

/* ===================================================================== */
void PrintResults(void)
{
    ofstream out(KnobOutputFile.Value().c_str());

    out.setf(ios::fixed, ios::floatfield);
    out.precision(2);

    out << "---------------------------------------";
    out << endl << "\t\tSimulation Results" << endl;
    out << "---------------------------------------";

    out << endl
        << "Memory system->" << endl
        << "\t\tDcache size (bytes)         : " << dcache->getCacheSize() << endl
        << "\t\tDcache ways                 : " << dcache->getCacheAssoc() << endl
         << "\t\tDcache block size (bytes)   : " << dcache->getCacheBlockSize() << endl;

    out << endl
        << "\t\tIcache size (bytes)         : " << icache->getCacheSize() << endl
        << "\t\tIcache ways                 : " << icache->getCacheAssoc() << endl
        << "\t\tIcache block size (bytes)   : " << icache->getCacheBlockSize() << endl;

    out << endl
        << "\t\tL2-cache size (bytes)       : " << llcache->getCacheSize() << endl
        << "\t\tL2-cache ways               : " << llcache->getCacheAssoc() << endl
        << "\t\tL2-cache block size (bytes) : " << llcache->getCacheBlockSize() << endl;

    if (vc) {
        out << endl
            << "\t\tVD-cache size (bytes)       : " << vc->getCacheSize() << endl;
    }

    if (pb) {
        out << endl
            << "\t\tPD-cache size (bytes)       : " << pb->getCacheSize() << endl;
    }

    if (ifcache) {
        out << endl
            << "\t\tI-Filter cache size (bytes)       : " << ifcache->getCacheSize() << endl
            << "\t\tI-Filter cache ways               : " << ifcache->getCacheAssoc() << endl
            << "\t\tI-Filter cache block size (bytes) : " << ifcache->getCacheBlockSize() << endl;
    }
    if (dfcache) {
        out << endl
            << "\t\tD-Filter cache size (bytes)       : " << dfcache->getCacheSize() << endl
            << "\t\tD-Filter cache ways               : " << dfcache->getCacheAssoc() << endl
            << "\t\tD-Filter cache block size (bytes) : " << dfcache->getCacheBlockSize() << endl;
    }

    out << endl;


    out << "Simulated events->" << endl;

    out << "\t\t I-Cache Miss: " << icache->getTotalMiss() << " out of " << icache->getRequest() << endl;

    out << "\t\t D-Cache Miss: " << dcache->getTotalMiss() << " out of " << dcache->getRequest() << endl;

    out << "\t\t L2-Cache Miss: " << llcache->getTotalMiss() << " out of " << llcache->getRequest() << endl;

    if (vc) out << "\t\t VD-Cache Miss: " << vc->getTotalMiss() << " out of " << vc->getRequest() << endl;

    if (pb) out << "\t\t PD-Cache Miss: " << pb->getTotalMiss() << " out of " << pb->getRequest() << endl;

    if (ifcache) out << "\t\t I-Filter Cache Miss: " << ifcache->getTotalMiss() << " out of " << ifcache->getRequest() << endl;

    if (dfcache) out << "\t\t D-Filter Cache Miss: " << dfcache->getTotalMiss() << " out of " << dfcache->getRequest() << endl;
    
    out << endl;

    out << "\t\t Requests resulted in " << icache->getRequest() + dcache->getRequest() << " L1 requests, "
        << llcache->getRequest() << " L2 requests, "
        << (*mem).getRequest() << " mem requests " << endl;

    out << endl;

    out << endl;
    out << "------------------------------------------------";
    out << endl;
}

/// Add instruction instrumentation calls
/* ===================================================================== */
void Instruction(INS ins, VOID *v)
{
    // All instructions access the icache
    INS_InsertCall(ins, IPOINT_BEFORE, (AFUNPTR) AllInstructions,
            IARG_INST_PTR, IARG_END);

    // Reads go to dcache
    if (INS_IsMemoryRead(ins)) {
        INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE, (AFUNPTR) MemoryOp,
                IARG_INST_PTR,
                IARG_MEMORYREAD_EA,
                IARG_END);

    }

    // Writes go to dcache
    // XXX: note this is not an else branch. It's pretty typical for an x86
    // instruction to be both a read and a write.
    if ( INS_IsMemoryWrite(ins) ) {
        INS_InsertPredicatedCall(
                ins, IPOINT_BEFORE,  (AFUNPTR) MemoryOp,
                IARG_INST_PTR,
                IARG_MEMORYWRITE_EA,
                IARG_END);
    }
}

/* ===================================================================== */
void Fini(int n, VOID *v)
{
    PrintResults();
}

/* ===================================================================== */
int main(int argc, char *argv[])
{
    if( PIN_Init(argc,argv) ){
        return Usage();
    }

    CreateCaches();

    INS_AddInstrumentFunction(Instruction, 0);

    PIN_AddFiniFunction(Fini, 0);

    PIN_StartProgram();

    return 0;
}
