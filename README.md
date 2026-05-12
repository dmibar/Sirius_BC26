# Система для автоматической проверки статистических гипотез
Этот репозиторий содержит код для системы на [LangGraph](https://www.langchain.com/langgraph), которая способна за **3-4** секунды проверять статистические гипотезы, на основе табличных данных.\
Используя проверки условий применения, система подберет статистический тест, и проверит гипотезу. После, предоставит сводку исследования, в которой указаны все этапы анализа и вывод.


Проект стал призером международного конкурса научно-технологических проектов ['Большие вызовы'](https://bigchallenges.ru/) в [Сириусе](https://sochisirius.ru/).
# Граф работы системы
```mermaid
graph TD
    Start([__start__]) --> FindVars[Поиск переменных]
    FindVars --> Wait[Ожидание ввода]
    Wait --> IsCorrect{Переменные корректны?}
    
    IsCorrect -- Нет --> FindVars
    IsCorrect -- Да --> FixTypes[Корректировка типов данных]
    
    FixTypes --> Preprocess[Предобработка]
    Preprocess -- Переменные присутствуют в данных --> CleanTarget{Выбор стат теста}
    
    Preprocess -- Полученных значений нет в данных --> End

    CleanTarget -- "Одна количественная другая номинативная" --> NormDist[Проверка нормальности]
    CleanTarget -- "Много групп и количественный таргет" --> NormDistVar[Проверка нормальности групп]
    CleanTarget -- "Все категориальные" --> Chi2[Критерий Хи-квадрат]
    CleanTarget -- "Все числовые" --> LinReg[Линейная регрессия]
    CleanTarget -- "Много групп и категориальный таргет" --> LogitReg[Логистическая регрессия]

    NormDist --> MakeSamples[Подготовка выборок]
    MakeSamples --> MW_Test{Выбор теста}
    
    MW_Test -- "Манна-Уитни" --> MW_Sub{Вариант}
    MW_Sub -- "Бакетированный" --> MW_Bake[Манна-Уитни - Бакетированный]
    MW_Sub -- "Стандарт" --> MW_Std[Манна-Уитни - Std]
    
    MW_Test -- "T-test" --> TTest[T-критерий Стьюдента]
    MW_Test -- "Welch" --> Welch[T-критерий Уэлча]

    NormDistVar --> ChooseStat[Выбор статистики]
    ChooseStat --> VarAnalysis[Дисперсионный анализ]
    VarAnalysis --> LinVarReg[Линейная регрессия]

    Chi2 --> Fisher[Точный критерий Фишера]
    LinReg --> Corrs[Корреляционный анализ]

    MW_Bake & MW_Std & TTest & Welch & LinVarReg & Fisher & Corrs & LogitReg --> Summary[Сводный отчет LLM]
    Summary --> End([__end__])
```

# Пример работы системы
![Example](./images/img.png)

# Инструкция по установке и запуску

### 1. Клонирование репозитория
Склонируйте проект на локальный компьютер и перейдите в рабочую директорию:
```bash
git clone https://github.com
cd Sirius_BC26
```

### 2. Создание виртуального окружения
Создайте изолированное окружение, чтобы избежать конфликтов библиотек:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**MacOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Установка зависимостей и настройка ядра
Установите необходимые пакеты и зарегистрируйте окружение в Jupyter:
```bash
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Python (Sirius_BC26)"
```

### 4. Запуск
1. Откройте ноутбук [CodeOfSystem.ipynb](Code/CodeOfSystem.ipynb) через VS Code или Jupyter Lab.
2. В интерфейсе выбора ядра (обычно в правом верхнем углу) выберите **"Python (Sirius_BC26)"**.
3. Теперь вы можете запускать ячейки кода.


# Структура проекта
```mermaid
flowchart TD
    A[AutoStatFlow] --> B(Code)
    A --> C(Docs)
    A --> D(research)

    B --> a(Весь код системы)
    C --> b(Документация)
    D --> d(Исследования)
```
## Лицензия
Этот проект распространяется под лицензией MIT.


# Automated Statistical Hypothesis Testing System

This repository contains the code for a [LangGraph](https://langchain.com)-based system capable of testing statistical hypotheses on tabular data in just **3–4 seconds**.

By evaluating application assumptions and requirements, the system automatically selects the appropriate statistical test and performs the analysis. It then provides a comprehensive research summary, detailing every stage of the analysis and the final conclusion.

This project was a prize-winner at the ['Big Challenges'](https://bigchallenges.ru) international scientific and technological project competition held at [Sirius](https://sochisirius.ru).
