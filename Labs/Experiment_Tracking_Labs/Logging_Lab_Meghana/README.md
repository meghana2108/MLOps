# Logging Lab — Meghana

A walkthrough of Python's built-in `logging` library, based on the course's `Logging_Labs/Starter.ipynb`, with fixes to the starter code and extensions that apply logging to an ML training loop. Uses only the Python standard library.

## Files

- `Logging_Lab_Meghana.ipynb` — the completed notebook
- `logs/` — log files written when the notebook runs (`app.log`, `my_module.log`, rotating logs, and one log per training run)

## What the notebook covers

1. Basic configuration with `basicConfig`
2. The five log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
3. Named loggers instead of the root logger
4. Logging exceptions with tracebacks using `logger.exception()`
5. Logging to a file
6. Changing log levels at runtime
7. Handlers: console and file, each with its own level and format

## Fixes to the starter notebook

- **Logging to a file didn't work.** The second `basicConfig(filename='app.log', ...)` call did nothing because the root logger was already configured. Added `force=True` and read the file back to show it's written.
- **Messages printed twice.** The custom logger's handler printed each message, then the message also propagated to the root logger and printed again. Set `propagate = False`.
- **Handlers had no formatter**, so one copy of each message had no timestamp or level. Each handler now gets a formatter.
- **Re-running the handler cell added duplicate handlers.** The cell now clears existing handlers first, so it's safe to re-run.
- Fixed the "noccurred" typo, and log files now go into a `logs/` folder.

## Extensions

- **Different levels per handler:** console shows INFO and above, the file keeps DEBUG too.
- **Rotating log files** with `RotatingFileHandler`, so logs don't grow without limit.
- **Structured JSON logs** with a custom `Formatter`, including extra fields like `epoch` and `loss`, which is the format log platforms like ELK work best with.
- **Logging an ML training run:** a small linear regression trained with gradient descent, logging progress at INFO, every epoch at DEBUG (file only), a WARNING whenever the loss increases, and an ERROR with traceback when training diverges. It's run once with a good learning rate (converges to w ≈ 3, b ≈ 2) and once with one that's too high (warnings, then stopped with an error at epoch 4).

## How to run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install notebook
jupyter notebook Logging_Lab_Meghana.ipynb
```

Then run all cells.
