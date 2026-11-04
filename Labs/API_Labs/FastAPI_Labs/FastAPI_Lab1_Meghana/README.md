# FastAPI Lab 1 – Serving an Iris Classifier as an API

- Video Explanation: [FastAPI lab](https://www.youtube.com/watch?v=KReburHqRIQ&list=PLcS4TrUUc53LeKBIyXAaERFKBJ3dvc9GZ&index=4)
- Blog: [FastAPI Lab-1](https://www.mlwithramin.com/blog/fastapi-lab1)

---

## Overview

In this lab, a machine learning model is exposed as an API using [FastAPI](https://fastapi.tiangolo.com/) and [uvicorn](https://www.uvicorn.org/).

1. **FastAPI**: a modern, high-performance web framework for building APIs with Python based on standard Python type hints.
2. **uvicorn**: an [ASGI](https://youtu.be/vKjCkeJGbNk) web server implementation for Python, commonly used to serve FastAPI applications.

The workflow:

1. Train a Decision Tree Classifier on the Iris dataset.
2. Serve the trained model as an API using FastAPI and uvicorn.

On top of the base lab, this version adds species names with confidence scores, input validation, error handling, a health check endpoint, and a batch prediction endpoint (see [My Modifications](#my-modifications)).

---

## Setting up the lab

1. Create a virtual environment:
   ```bash
   python3 -m venv fastapi_lab1_env
   ```
2. Activate it:
   ```bash
   source fastapi_lab1_env/bin/activate      # Mac/Linux
   fastapi_lab1_env\Scripts\activate         # Windows
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

Note: **fastapi[all]** in **requirements.txt** installs optional extra dependencies for FastAPI, including **uvicorn**.

### Project structure

```
FastAPI_Labs
└── FastAPI_Lab1_Meghana
    ├── fastapi_lab1_env/        # virtual environment (not committed)
    ├── model/
    │   └── iris_model.pkl
    ├── src/
    │   ├── __init__.py
    │   ├── data.py              # loads and splits the Iris dataset
    │   ├── train.py             # trains, evaluates, and saves the model
    │   ├── predict.py           # loads the model and returns predictions
    │   └── main.py              # FastAPI app and endpoints
    ├── .gitignore
    ├── README.md
    └── requirements.txt
```

---

## Running the Lab

1. Move into the **src/** folder:
   ```bash
   cd src
   ```
2. Train the Decision Tree Classifier (this overwrites `model/iris_model.pkl` with a fresh model):
   ```bash
   python train.py
   ```
   Expected output:
   ```
   Test accuracy: 0.956
   Model saved to .../model/iris_model.pkl
   ```
3. Serve the model as an API:

   ```bash
   uvicorn main:app --reload
   ```

   > Note: the original README had a typo (`uvicorn app:main --reload`). The correct command is `uvicorn main:app --reload`.

4. Open the interactive docs at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (or [http://localhost:8000/docs](http://localhost:8000/docs)).

5. To test an endpoint: click on it → **Try it out** → fill in the request body → **Execute**. Tools like [Postman](https://www.postman.com/) also work.

---

## API Endpoints

| Method | Endpoint         | Description                                   |
| ------ | ---------------- | --------------------------------------------- |
| GET    | `/`              | Health check, returns `{"status": "healthy"}` |
| POST   | `/predict`       | Predicts the species for one flower           |
| POST   | `/predict/batch` | Predicts the species for up to 100 flowers    |

### Example: `POST /predict`

Request:

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

Response:

```json
{ "response": 0, "species": "setosa", "confidence": 1.0 }
```

### Example: `POST /predict/batch`

Request:

```json
[
  {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  },
  {
    "sepal_length": 6.7,
    "sepal_width": 3.0,
    "petal_length": 5.2,
    "petal_width": 2.3
  }
]
```

Response: a list with one prediction per flower (setosa and virginica for the example above).

### Example: invalid input

Sending `"petal_length": -3` returns a **422 Unprocessable Entity** error, because every measurement must be greater than 0 and less than 10 cm.

---

## My Modifications

- **Species name and confidence**: the prediction response now includes the species name and the model's confidence (via `predict_proba`), not just the class number.
- **Input validation**: added Pydantic `Field` constraints so every measurement must be between 0 and 10 cm; invalid inputs return a 422 error.
- **Error handling**: added `HTTPException` responses (503 if the model file is missing, 500 if prediction fails, 400 for empty or oversized batches).
- **Health check**: added a `GET /` endpoint.
- **Batch prediction**: added a `POST /predict/batch` endpoint that accepts up to 100 samples in one request.
- **Training output**: `train.py` now prints test accuracy and the model save path.
- **Bug fix**: corrected the run command typo to `uvicorn main:app --reload`.

---

## FastAPI Concepts Used

### FastAPI syntax

- The app instance is created with:
  ```python
  app = FastAPI(title="Iris Classifier API", version="1.0")
  ```
- In `uvicorn main:app --reload`, **main** is the Python file containing the app (without `.py`) and **app** is the FastAPI instance. **--reload** restarts the server when code changes. It is useful during development but should not be used in production.
- Route handlers are defined with decorators such as `@app.get("/")` and `@app.post("/predict")`:
  1. **Decorator (@)**: attaches a function to an HTTP method and path.
  2. **App instance (app)**: holds the routes, handlers, and configuration.
  3. **HTTP method**: `get` retrieves data, `post` sends data. FastAPI also supports `put`, `delete`, `patch`, and `options` ([MDN reference](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods)).
  4. **Path**: the URL where the endpoint is accessible.
- **async** route handlers allow non-blocking I/O, so the server can handle other requests while waiting on tasks like model loading.

### Data models

**Request model, `IrisData`:**

```python
class IrisData(BaseModel):
    sepal_length: float = Field(..., gt=0, lt=10, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, lt=10, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, lt=10, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, lt=10, description="Petal width in cm")
```

When used as a route parameter, FastAPI reads the JSON body, converts types (for example, `"5.1"` → `5.1`), and validates the data. Missing fields, wrong types, or values outside 0–10 return a **422** error with details.

**Response model, `IrisResponse`:**

```python
class IrisResponse(BaseModel):
    response: int
    species: str
    confidence: float
```

Setting `response_model=IrisResponse` makes FastAPI serialize the output to match this structure and documents it on the `/docs` page.

### Error handling

`HTTPException` returns an explicit status code and a message:

```python
try:
    return predict_data([to_features(iris)])[0]
except FileNotFoundError as e:
    raise HTTPException(status_code=503, detail=str(e))
```

If the model file is missing, the client receives:

```json
{ "detail": "Model file not found. Run train.py first." }
```

### Screenshots:

<img width="1497" height="759" alt="image" src="https://github.com/user-attachments/assets/e87fbb8f-c97a-4077-b5f4-70ce51312c3d" />

<img width="1163" height="775" alt="image" src="https://github.com/user-attachments/assets/76369af3-4945-4b54-8a8d-1f394f0d6cc3" />



For more, see the [FastAPI error handling docs](https://fastapi.tiangolo.com/tutorial/handling-errors/).
