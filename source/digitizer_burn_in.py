#!/usr/bin/env python

import datetime
import json
import time

import dripline

interface = dripline.core.Interface("rabbit_broker", "burn-in-requester")

# run_data are json-encoded and passed to fast_daq as the run's "description"
run_data = {
            "flux_cap": 7.2,
            "run_id": '?',
            "notes": "completed a burn-in test run",
           }

was_running = False

while True:
    while True:
        result = interface.get('psyllid.daq-status')['payload']['server']
        print("{}| run <{}> status is [{}]: {}".format(datetime.datetime.now(), run_data['run_id'], result['status-value'], result['status']))
        if result['status-value'] == 4:
            if was_running:
                print("run {} complete, sleeping".format(run_data['run_id']))
                time.sleep(20)
            was_running = False
            break
        elif result['status-value'] == 5:
            was_running = True
            time.sleep(5)
            continue
        else:
            raise ValueError("daq is not in an understood status")
    # start a run after waiting to ensure the run is over, because sometimes things crash/restart while the other is still going fine
    insert_result = interface.cmd("spectrum_table", "do_insert", **{"notes":"starting a burn-in test run", 'timestamp': datetime.datetime.utcnow().isoformat()+'+00'})
    print("the new run_id *should* be [{}]".format(insert_result['payload']['digitizer_log_id']))
    run_data['run_id'] = insert_result['payload']['digitizer_log_id']
    this_filename = "/data/{}.egg".format(run_data['run_id'])
    print("{} - starting run number {}".format(datetime.datetime.now(), run_data['run_id']))
    interface.cmd('psyllid', 'start-run', duration=0, filename=this_filename, description=json.dumps(run_data))
    time.sleep(5)
