#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
require MODEL_PATH GATEWAY_HOST
run "$VENV/bin/python" -m smg launch --pd-disaggregation --backend tokenspeed \
 --prefill "tcp://$GATEWAY_HOST:18510" --decode "tcp://$GATEWAY_HOST:18511" \
 --model-path "$MODEL_PATH" --tokenizer-path "$MODEL_PATH" --model-alias "deepseek-v41-flash=$MODEL_PATH" \
 --host "$GATEWAY_HOST" --port 18500 --prefill-policy round_robin --decode-policy round_robin \
 --max-concurrent-requests 32 --queue-size 128 --queue-timeout-secs 7200 --request-timeout-secs 7200 \
 --disable-retries --disable-circuit-breaker --prometheus-port 18520 \
 --worker-stall-secs 1800 --worker-wedge-secs 1800 --worker-stale-secs 1800 \
 --grpc-request-timeout-secs 7200 --max-payload-size 33554432 --max-buffered-request-bytes 33554432 "$@"
