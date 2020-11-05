
# genderBERT

TEXT

```mermaid
graph LR
A(review texts) -- + author name --> B[gender detector]
B --> C(train dataset)
B --> D(test dataset)
C -- training --> F[genderBERT]
F -- majority voting --> F
D -- testing --> F


```

## Statistics

### Without majority-voting
| Model | Male F1 | Female F1 | Accuracy  |
|---|---|---|---|
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |

### With majority voting
| Model | Male F1 | Female F1 | Accuracy  |
|---|---|---|---|
|   |   |   |   |
|   |   |   |   |
|   |   |   |   |
