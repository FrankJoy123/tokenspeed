// Copyright (c) 2026 LightSeek Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Compatibility at the transport boundary for the gateway's control RPC.
use bytes::Bytes;
use ts_wire::{decode_frames, encode_frames, ControlRequest, EngineCoreReadyResponse, EngineRequest, Value};
use ts_wire::control::FlushCacheReqInput;
use ts_wire::msgpack::Integer;

pub(super) enum Input {
    Request((EngineRequest, Option<i64>)),
    Reply(Vec<Bytes>),
}

pub(super) fn ready_frames(ready: &EngineCoreReadyResponse) -> Vec<Bytes> {
    // Add a capability only on this external transport; native clients keep
    // their existing tagged control schema and reply handling.
    let frames=encode_frames(ready);
    let Value::Map(mut fields)=decode_frames::<Value>(&frames).expect("ready map") else { unreachable!() };
    fields.push((Value::Str("positional_layouts".into()), Value::Map(vec![(
        Value::Str("ControlReplySlim".into()),
        Value::Array(["call_id", "success", "message", "engine_index"].into_iter().map(|s| Value::Str(s.into())).collect()),
    )])));
    encode_frames(&Value::Map(fields))
}

pub(super) fn reply(id: i64, success: bool, message: Option<String>, engine: u16) -> Vec<Bytes> {
    encode_frames(&Value::Array(vec![Value::Str("ControlReplySlim".into()),
        Value::Int(Integer::Signed(id)), Value::Bool(success),
        message.map_or(Value::Nil, Value::Str), Value::Int(Integer::Unsigned(u64::from(engine)))]))
}

pub(super) fn decode(frames: &[Bytes], engine: u16) -> ts_wire::Result<Input> {
    // Native tagged controls and all data-plane messages retain their path.
    if let Ok(request)=EngineRequest::from_frames(frames) {
        return Ok(Input::Request((request,None)));
    }
    if frames.len()!=2 || frames[0].as_ref()!=[ts_wire::frames::REQ_TYPE_CONTROL] {
        return EngineRequest::from_frames(frames).map(|r| Input::Request((r,None)));
    }
    let value=decode_frames::<Value>(&frames[1..])?;
    let Value::Array(fields)=value else { return Err(ts_wire::Error::Invalid("control call must be an array".into())); };
    let [Value::Int(id),Value::Str(method),Value::Array(args)]=fields.as_slice() else {
        return Err(ts_wire::Error::Invalid("expected (call_id, method, args)".into()));
    };
    let id=(*id).narrow::<i64>("call_id")?;
    // Flush is nonblocking: the scheduler returns false if it cannot clear
    // safely. Do not silently promise the gateway's optional wait semantics.
    let immediate=args.is_empty() || matches!(args.as_slice(), [Value::Float(t)] if *t==0.0)
        || matches!(args.as_slice(), [Value::Int(t)] if t.narrow::<i64>("timeout").ok()==Some(0));
    if method!="flush_cache" || !immediate {
        return Ok(Input::Reply(reply(id,false,Some("Only immediate flush_cache (timeout_s=0) is supported by this control adapter".into()),engine)));
    }
    Ok(Input::Request((EngineRequest::Control(Box::new(ControlRequest::FlushCache(FlushCacheReqInput {rid:None,http_worker_ipc:None}))),Some(id))))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn call(method: &str, timeout: f64) -> Vec<Bytes> {
        let mut f=vec![Bytes::from_static(&[2])];
        f.extend(encode_frames(&Value::Array(vec![Value::Int(Integer::Signed(17)),Value::Str(method.into()),Value::Array(vec![Value::Float(timeout)])])));f
    }
    #[test]
    fn gateway_flush_enters_the_native_scheduler_path() {
        let Input::Request((EngineRequest::Control(op),Some(17)))=decode(&call("flush_cache",0.0),7).unwrap() else { panic!("not a native flush") };
        assert!(matches!(op.as_ref(),ControlRequest::FlushCache(_)));
    }
    #[test]
    fn native_flush_is_not_reencoded_as_a_gateway_call() {
        let request=EngineRequest::Control(Box::new(ControlRequest::FlushCache(FlushCacheReqInput{rid:None,http_worker_ipc:None})));
        let Input::Request((actual,None))=decode(&request.to_frames(),0).unwrap() else { panic!("native control lost") };
        assert_eq!(actual,request);
    }
    #[test]
    fn unsupported_calls_reply_with_failure_and_correlation() {
        for (method,timeout) in [("start_profile",0.0),("flush_cache",1.0)] {
            let Input::Reply(frames)=decode(&call(method,timeout),7).unwrap() else { panic!("unsupported call admitted") };
            let Value::Array(fields)=decode_frames::<Value>(&frames).unwrap() else { panic!() };
            assert_eq!(fields[2],Value::Bool(false));
            assert_eq!(fields[1],Value::Int(Integer::Unsigned(17)));
            assert_eq!(fields[4],Value::Int(Integer::Unsigned(7)));
        }
    }
}
