# Design contracts

These documents define implementation ownership and invariants. Change the
relevant contract with a change to the implementation it describes.

- [Unified execution](unified_path.md): eager/captured execution, padding,
  QCP row ownership and native operator inputs.
- [Attention prologue](attention-prologue.md): projection, norm, RoPE,
  quantization and cache publication.
- [Cache concepts](cache-concepts.md) and [scheduler](scheduler.md): persistent
  state, admission and lifetime.
- [Event loop](event-loop.md): control/data plane responsibilities.
- [SM120 full-model paths](sm120.md): platform-specific transport and
  quantization contracts extending the common execution path.
