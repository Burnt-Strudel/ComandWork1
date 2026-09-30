import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Прогноз стоимости недвижимости",
    page_icon="🏠",
    layout="wide",
)


def fmt_num(x: float) -> str:
    """100000 → '100.000'"""
    return f"{x:,.0f}".replace(",", ".")


@st.cache_resource
def load_artifacts():
    root = Path("models")
    candidates = {
        "model": [
            root / "best_model_111.pkl",
            root / "best_model_111.joblib",
            Path("best_model_111.pkl"),
            Path("best_model_111.joblib"),
        ],
        "features": [root / "feature_columns.pkl", Path("feature_columns111.pkl")],
        "scaler": [root / "scaler.pkl", Path("scaler111.pkl")],
        "scale_cols": [root / "scale_cols.pkl", Path("scale_cols111.pkl")],
    }

    def first_existing(paths):
        for p in paths:
            if p.exists():
                return p
        return None

    model_path = first_existing(candidates["model"])
    features_path = first_existing(candidates["features"])
    scaler_path = first_existing(candidates["scaler"])
    scale_cols_path = first_existing(candidates["scale_cols"])

    if model_path is None:
        raise FileNotFoundError(
            "Не найден best_model.pkl / .joblib (папка models/ или рядом с app.py)"
        )
    if features_path is None:
        raise FileNotFoundError(
            "Не найден feature_columns.pkl — без него нельзя собрать вектор признаков"
        )

    model = joblib.load(model_path)
    feature_columns = joblib.load(features_path)
    scaler = joblib.load(scaler_path) if scaler_path else None
    scale_cols = joblib.load(scale_cols_path) if scale_cols_path else []

    return {
        "model": model,
        "feature_columns": feature_columns,
        "scaler": scaler,
        "scale_cols": scale_cols or [],
        "model_path": str(model_path),
    }


try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(f"Не удалось загрузить модель: {e}")
    st.stop()

model = artifacts["model"]
feature_columns = artifacts["feature_columns"]
scaler = artifacts["scaler"]
scale_cols = artifacts["scale_cols"]

# MAE Random Forest на test
MAE_TRY = 107_012

# ---------------------------------------------------------------------------
# Значения формы
# ---------------------------------------------------------------------------
SUB_TYPES = [
    "Квартира", "Резиденция", "Вилла", "Отдельный дом",
    "Дача", "Сборный дом", "Целое здание", "Особняк / Усадьба / Дом у воды",
    "Фермерский дом", "Лофт", "Сельский дом", "Другие",
]

CITIES = [
    "Стамбул", "Анкара", "Измир", "Анталья", "Бурса",
    "Мерсин", "Айдын", "Мугла", "Адана", "Конья",
    "Другие",
]

HEATING = [
    "Фанкойл", "Центральное (газ)", "Газовый котёл", "Центральное",
    "Центральное (счётчик тепла)", "Кондиционер", "Печь (уголь)",
    "Печь (дрова)", "Печь", "Тёплый пол", "Нет", "Не указано", "Другие",
]

BUILDING_AGE_OPTS = [
    "0", "1", "2", "3", "4", "5",
    "6-10 лет", "11-15 лет", "16-20 лет", "21-25 лет",
    "26-30 лет", "31-35 лет", "36-40 лет", "40 и более",
]

LISTING_TYPES = ["Продажа", "Аренда", "Аренда на день"]

AGE_ORDER = [
    "0", "1", "2", "3", "4", "5",
    "6-10 лет", "11-15 лет", "16-20 лет", "21-25 лет",
    "26-30 лет", "31-35 лет", "36-40 лет", "40 и более", "Другие",
]
AGE_MAP = {v: i for i, v in enumerate(AGE_ORDER)}


def build_feature_row(
    total_area, rooms_num, floor_num, floors_total_num, tom,
    building_age, sub_type, city, heating_type, listing_type,
):
    """Вектор признаков в порядке feature_columns (числовые + one-hot)."""
    row = {col: 0.0 for col in feature_columns}

    numeric_map = {
        "total_area": float(total_area),
        "rooms_num": float(rooms_num),
        "floor_num": float(floor_num),
        "floors_total_num": float(floors_total_num),
        "tom": float(tom),
        "building_age": float(AGE_MAP.get(building_age, len(AGE_ORDER))),
    }
    for k, v in numeric_map.items():
        if k in row:
            row[k] = v

    def set_one_hot(prefix, value):
        target = f"{prefix}_{value}"
        if target in row:
            row[target] = 1.0
            return
        fallback = f"{prefix}_Другие"
        if fallback in row:
            row[fallback] = 1.0

    set_one_hot("sub_type", sub_type)
    set_one_hot("city", city)
    set_one_hot("heating_type", heating_type)
    set_one_hot("listing_type", listing_type)

    X = pd.DataFrame([row], columns=feature_columns)

    if scaler is not None and scale_cols:
        cols = [c for c in scale_cols if c in X.columns]
        if cols:
            X = X.copy()
            X[cols] = scaler.transform(X[cols])

    return X


def predict_price(X):
    """Модель на log_price → цена в TRY."""
    pred_log = float(model.predict(X)[0])
    return max(0.0, float(np.expm1(pred_log)))


# ---------------------------------------------------------------------------
# Навигация
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Раздел",
    ["Прогноз стоимости", "Дашборд", "Справка"],
)
st.sidebar.caption(f"Модель: `{artifacts['model_path']}`")

# ===========================================================================
# 4.1 ФОРМА ПРОГНОЗА
# ===========================================================================
if page == "Прогноз стоимости":
    st.title("Прогноз стоимости недвижимости")
    st.write(
        "Заполните характеристики объекта и нажмите **«Рассчитать»**. "
        "Прогноз в **турецких лирах (TRY)**."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Основные")
        listing_type = st.selectbox("Тип объявления", LISTING_TYPES)
        sub_type = st.selectbox("Тип жилья", SUB_TYPES, index=0)
        total_area = st.number_input(
            "Площадь, м²", min_value=10.0, max_value=2000.0, value=90.0, step=5.0
        )
        rooms_num = st.number_input(
            "Число комнат (всего)", min_value=1.0, max_value=20.0, value=3.0, step=0.5
        )

    with col2:
        st.subheader("Здание")
        building_age = st.selectbox("Возраст здания", BUILDING_AGE_OPTS, index=5)
        floors_total_num = st.number_input(
            "Этажность дома", min_value=1.0, max_value=50.0, value=10.0, step=1.0
        )
        floor_num = st.number_input(
            "Этаж объекта", min_value=0.0, max_value=50.0, value=3.0, step=1.0
        )
        heating_type = st.selectbox("Отопление", HEATING, index=0)

    with col3:
        st.subheader("Локация и прочее")
        city = st.selectbox("Город", CITIES, index=0)
        tom = st.number_input(
            "Дней на рынке (tom)", min_value=0, max_value=500, value=30, step=1
        )
        st.caption("Район/микрорайон в модели не используются (только city).")

    errors = []
    if floor_num > floors_total_num:
        errors.append("Этаж объекта не может быть больше этажности дома.")

    if st.button("Рассчитать стоимость", type="primary", use_container_width=True):
        if errors:
            for e in errors:
                st.error(e)
        else:
            try:
                X = build_feature_row(
                    total_area=total_area,
                    rooms_num=rooms_num,
                    floor_num=floor_num,
                    floors_total_num=floors_total_num,
                    tom=tom,
                    building_age=building_age,
                    sub_type=sub_type,
                    city=city,
                    heating_type=heating_type,
                    listing_type=listing_type,
                )
                pred = predict_price(X)
                low = max(0.0, pred - MAE_TRY)
                high = pred + MAE_TRY

                st.success("Прогноз готов")
                m1, m2, m3 = st.columns(3)
                m1.metric("Точечный прогноз", f"{fmt_num(pred)} ₺")
                m2.metric("Нижняя граница (−MAE)", f"{fmt_num(low)} ₺")
                m3.metric("Верхняя граница (+MAE)", f"{fmt_num(high)} ₺")

                st.info(
                    f"Диапазон ≈ ±MAE (**{fmt_num(MAE_TRY)} ₺**). "
                    "Ориентир по средней ошибке на тесте, не строгий доверительный интервал. "
                    "В данных есть продажа и аренда — проверяйте «Тип объявления»."
                )
            except Exception as e:
                st.error(f"Ошибка расчёта: {e}")
                st.exception(e)

# ===========================================================================
# 4.2 ДАШБОРД
# ===========================================================================
elif page == "Дашборд":
    st.title("Дашборд")

    DASHBOARD_URL = "https://datalens.yandex/24jq7mf50liml?_share_link=public"

    st.link_button("Открыть дашборд DataLens", DASHBOARD_URL, type="primary")
    st.markdown(f"[Ссылка на дашборд]({DASHBOARD_URL})")

# ===========================================================================
# 4.3 СПРАВКА
# ===========================================================================
else:
    st.title("Справка")
    st.markdown(
        """
### Что делает приложение
Прогнозирует **цену объекта в турецких лирах (TRY)**
по характеристикам с помощью обученной модели (Random Forest).

### Как пользоваться
1. Вкладка **«Прогноз стоимости»**
2. Заполни поля
3. **«Рассчитать стоимость»**
4. Точечный прогноз и диапазон ±MAE

### Описание полей

| Поле | Смысл | Единицы |
| --- | --- | --- |
| Тип объявления | Продажа / Аренда / Аренда на день | категория |
| Тип жилья | Квартира, Вилла, Резиденция… | категория |
| Площадь | Общая площадь | м² |
| Число комнат | `rooms_num` | шт. |
| Возраст здания | Диапазон возраста | категория → порядковый код |
| Этажность / этаж | Этажей в доме и этаж объекта | шт. |
| Отопление | Тип системы | категория |
| Город | Локация | топ-города + «Другие» |
| tom | Дней на рынке | дни |

### Модель и ограничения
- **Модель:** Random Forest (R² ≈ 0.966, MAE ≈ 107.012 ₺, MAPE ≈ 30.8%)
- Обучение на **log(price)**; в UI — обратное преобразование `expm1`
- **Валюты при подготовке данных приведены к TRY**
- В выборке смешаны продажа и аренда
- Редкие категории схлопывались в «Другие»
- ±MAE — эвристика, не статистический ДИ

### Технические детали
- `models/best_model.pkl`
- `models/feature_columns.pkl`
- `models/scaler.pkl` + `models/scale_cols.pkl`
- Загрузка один раз: `@st.cache_resource`
- **preprocessor не используется**

### Версия
- 1.1 | Python, scikit-learn, Streamlit, joblib
"""
    )