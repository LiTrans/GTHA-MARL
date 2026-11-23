def clear_unregistered_employed_search(all_workers, employed_searchers):
    registered = set(employed_searchers)
    stale = []
    for worker_id, state in all_workers.items():
        worker = state.worker if hasattr(state, "worker") else state
        if (
            worker.employment_status == "Employed"
            and getattr(worker, "is_searching", False)
            and (worker_id not in registered)
        ):
            worker.is_searching = False
            stale.append(worker_id)
    return stale
