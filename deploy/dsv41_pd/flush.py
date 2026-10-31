"""Flush both idle PD workers without dropping JIT caches."""
import json
import os
import time


def validate_flush(result):
    if not (result.get('status') == 'success' and result.get('workers_flushed') == result.get('total_workers') == 2 and result.get('total_zmq_workers_skipped') == 0):
        raise ValueError(f'Incomplete PD cache flush: {result}')


def flush(url):
    import requests
    with requests.Session() as session:
        session.trust_env = False
        for _ in range(180):
            response = session.get(url + '/workers', timeout=10)
            response.raise_for_status()
            workers = response.json()['workers']
            if len(workers) == 2 and all(w['is_healthy'] for w in workers):
                loads = [load for w in workers for load in w.get('engine_load', {}).get('loads', [])]
                if loads and all(load.get('num_running_reqs', 0) == 0 and load.get('num_waiting_reqs', 0) == 0 for load in loads):
                    break
            time.sleep(2)
        else:
            raise TimeoutError('PD workers not confirmed idle; refusing to flush')
        response = session.post(url + '/flush_cache', timeout=120)
        response.raise_for_status()
        result = response.json()
        validate_flush(result)
        print(json.dumps(result))


if __name__ == '__main__':
    flush(os.environ['BENCH_URL'])
