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

//! Prefill bootstrap handshake compatibility.

use ts_wire::{EngineCoreReadyResponse, decode_bytes, decode_frames, encode_frames};

#[test]
fn ready_advertises_remote_prefill_bootstrap() {
    let ready = EngineCoreReadyResponse {
        bootstrap_host: Some("192.0.2.10".to_owned()),
        bootstrap_port: Some(18598),
        ..Default::default()
    };
    let bytes = encode_frames(&ready);
    let decoded: EngineCoreReadyResponse = decode_frames(&bytes).unwrap();
    assert_eq!(decoded.bootstrap_host.as_deref(), Some("192.0.2.10"));
    assert_eq!(decoded.bootstrap_port, Some(18598));
}

#[test]
fn legacy_ready_without_bootstrap_still_decodes() {
    let decoded: EngineCoreReadyResponse = decode_bytes(&[0x80]).unwrap();
    assert_eq!(decoded.bootstrap_host, None);
    assert_eq!(decoded.bootstrap_port, None);
}
