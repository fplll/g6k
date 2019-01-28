#ifndef G6K_HASH_TABLE_INL
#define G6K_HASH_TABLE_INL

#ifndef G6K_SIEVER_H
#error Do not include siever.inl directly
#endif

// resets the hash_function used. Siever is changed, because it uses it as a randomness source. This also clears the database.
// NOT THREAD-SAFE (including the randomness calls).
inline void UidHashTable::reset_hash_function(Siever& siever)
{
    n = siever.n;
    uid_coeffs.resize(n);
    for (unsigned i = 0; i < n; i++)
        uid_coeffs[i] = siever.rng.rng_nolock();
    for (unsigned i = 0; i < DB_UID_SPLIT; ++i)
        db_uid[i].clear();
    insert_uid(0);
    return;
}

// Compute the uid of x using the current hash function.
inline UidType UidHashTable::compute_uid(std::array<ZT,MAX_SIEVING_DIM> const &x) const
{
    ATOMIC_CPUCOUNT(250);
    return std::inner_product(x.cbegin(), x.cbegin()+n, uid_coeffs.cbegin(), static_cast<UidType>(0));
}

// resets the collision counter to 0. Returns its old value. NOT THEAD-SAFE
//inline unsigned long UidHashTable::reset_collision_counter()
//{
//  auto old_stat_C = stat_C;
//  stat_C = 0;
//  return old_stat_C;
//}

// atomically insert uid into the database
inline bool UidHashTable::insert_uid(UidType uid)
{
    ATOMIC_CPUCOUNT(251);
    normalize_uid(uid);
    std::lock_guard<std::mutex> lockguard(db_mut[uid % DB_UID_SPLIT]);
    bool success = db_uid[uid % DB_UID_SPLIT].insert(uid).second;
    if (success==false)
    {
//        STATS(++stat_C);
    }
    return success;
}

// check for presence of uid in the database.
inline bool UidHashTable::check_uid(UidType uid)
{
    ATOMIC_CPUCOUNT(252);
    normalize_uid(uid);
    std::lock_guard<std::mutex> lockguard(db_mut[uid % DB_UID_SPLIT]);
    if (db_uid[uid % DB_UID_SPLIT].count(uid) != 0)
    {
        return true;
    }
    else
        return false;
}

// atomically remove uid from the database.
inline bool UidHashTable::erase_uid(UidType uid)
{
    ATOMIC_CPUCOUNT(253);
    normalize_uid(uid);
    std::lock_guard<std::mutex> lockguard(db_mut[uid % DB_UID_SPLIT]);
    if (uid == 0)
        return false;
    return (db_uid[uid % DB_UID_SPLIT].erase(uid) != 0);
}

// atomically replace removed_uid by new_uid from the database.
inline bool UidHashTable::replace_uid(UidType removed_uid, UidType new_uid)
{
    ATOMIC_CPUCOUNT(254);
    if (UNLIKELY(removed_uid == 0)) // This should actually never happen anyway.
    {
        return false;
    }
    normalize_uid(removed_uid);
    normalize_uid(new_uid);

    // Note: We need to aquire both mutexes simultaneously.
    std::unique_lock<std::mutex> lock_new(db_mut[new_uid % DB_UID_SPLIT], std::defer_lock);
    std::unique_lock<std::mutex> lock_old(db_mut[removed_uid % DB_UID_SPLIT], std::defer_lock);
    if( (new_uid % DB_UID_SPLIT) == (removed_uid % DB_UID_SPLIT))
    {
        lock_new.lock();
    }
    else
    {
// Uses deadlock avoidance algorithm to simultaneously acquire both mutexes.
//        std::lock(lock_new, lock_old);
// We simplify by just always locking the smaller index first. -> This avoids deadlocks if done consistently.
        if( (new_uid % DB_UID_SPLIT) < (removed_uid % DB_UID_SPLIT))
        {
            lock_new.lock(); lock_old.lock();
        }
        else
        {
            lock_old.lock(); lock_new.lock();
        }
    }
    if (db_uid[new_uid % DB_UID_SPLIT].count(new_uid) != 0) // new uid is already present
    {
//        STATS(++stat_C);
        return false;
    }
    if (db_uid[removed_uid % DB_UID_SPLIT].erase(removed_uid) == 0) // try to erase removed_uid. On failure, we bail out.
    {
        return false;
    }
    db_uid[new_uid % DB_UID_SPLIT].insert(new_uid);
    return true;
}

// returns the number of stored hashes. NOT THREAD-SAFE
inline size_t UidHashTable::hash_table_size()
{
    assert(db_uid[0].count(0) == 1);
    size_t num_uid = 0;
    for (unsigned int batch = 0; batch < DB_UID_SPLIT; ++batch)
    {
        num_uid += db_uid[batch].size();
    }
    return num_uid - 1; // We do not count 0.
}


#endif
