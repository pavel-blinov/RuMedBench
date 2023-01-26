Решение реализовано с помощью модели [RuBioRoBERTa](https://huggingface.co/alexyalunin/RuBioRoBERTa).
Перед подачей в модель контекст и вопрос конкатенируются через пробел.

### Для модели были использованы гиперпараметры:
- `seed = 128`
- `batch_size = 16`
- `epochs = 25`
- `lr = 3e-5`

### Для запуска:
`pip install -r requirements.txt`

Открыть `RuMedNLI.ipynb` и выполнить все ячейки