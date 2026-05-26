# synth_eval

Biblioteca Python para avaliar dados sintéticos tabulares.

## Instalação / Uso

Nos notebooks, adicionar o path da pasta `datasets` ao início:

```python
import sys
sys.path.insert(0, "/Users/joaomonteiro/Desktop/Tese/datasets")
import synth_eval as se
```

---

## Estrutura do `train_datasets`

A biblioteca detecta automaticamente a estrutura:

```
train_datasets = {
    "real":  {seed: df, ...},
    "great": {llm: {seed: df, ...}, ...},   # LLM framework (nested)
    "ctgan": {seed: df, ...},               # GAN (flat)
}
```

---

## Módulos e funções

### `utils`
```python
se.get_column_types(df)                        # → (num_cols, cat_cols)
se.count_valid_rows(df)                        # → {total, valid, invalid, validity_rate}
se.preprocess_for_ml(df, label_encoders, fit)  # → (df_encoded, encoders)
se.run_validity_evaluation(train_datasets, methods, seeds)
```

### `utility`
```python
se.evaluate_utility(train_df, test_df, target, task='classification')
# task='classification' → {f1, auc, acc}
# task='regression'     → {rmse, mae, r2}

se.run_utility_evaluation(train_datasets, test_datasets, methods, seeds, target,
                           task='classification')
```

### `similarity`
```python
se.evaluate_discriminator(real_df, synthetic_df)         # → AUC
se.run_discriminator_evaluation(train_datasets, methods, seeds)

se.evaluate_correlation_difference(real_df, synthetic_df, method='pearson')  # → MAE
se.run_correlation_evaluation(train_datasets, methods, seeds, method='pearson')

se.calculate_percentage_equal_rows(df_real, df_synthetic)  # → %
se.run_percent_equal_rows_evaluation(train_datasets, methods, seeds)

se.calculate_percentage_similar_rows(df_real, df_synthetic, thresholds=[1.0,0.9,...])
se.run_percentage_similar_rows_evaluation(train_datasets, methods, seeds)
```

### `privacy`
```python
se.evaluate_dcr(real_df, synthetic_df)   # → {dcr_mean, dcr_std, dcr_min, dcr_5th, dcr_median}
se.run_dcr_evaluation(train_datasets, methods, seeds)

se.evaluate_mia(train_real_df, test_real_df, synthetic_df)  # → AUC
se.run_mia_evaluation(train_datasets, test_datasets, methods, seeds)
```

### `fairness`
```python
se.calculate_discrimination_score(df, sensitive_col, sensitive_condition, target, target_positive)
# → {ds, p_y1_s1, p_y1_s0, n_s1, n_s0}

se.run_fairness_evaluation(
    train_datasets, methods, seeds,
    sensitive_col='gender',
    sensitive_condition=lambda x: x == 'Male',
    target='income',
    target_positive='>50K'
)
```

---

## Exemplo completo

```python
import sys
sys.path.insert(0, "/Users/joaomonteiro/Desktop/Tese/datasets")
import synth_eval as se

# --- Carregar datasets (igual aos notebooks) ---
# train_datasets = { ... }
# test_datasets  = { ... }

methods = ['real', 'great']
seeds   = [42, 43, 44]
target  = 'income'

# Utility (classificação)
utility_df = se.run_utility_evaluation(train_datasets, test_datasets, methods, seeds, target)

# Discriminador
disc_df = se.run_discriminator_evaluation(train_datasets, methods, seeds)

# Correlação
corr_df = se.run_correlation_evaluation(train_datasets, methods, seeds)

# DCR
dcr_df = se.run_dcr_evaluation(train_datasets, methods, seeds)

# MIA
mia_df = se.run_mia_evaluation(train_datasets, test_datasets, methods, seeds)

# Fairness
fairness_df = se.run_fairness_evaluation(
    train_datasets, methods, seeds,
    sensitive_col='gender',
    sensitive_condition=lambda x: x == 'Male',
    target='income',
    target_positive='>50K'
)
```
