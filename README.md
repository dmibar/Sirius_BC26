# Система для автоматической проверки статистических гипотез
Этот репозиторий содержит код для системы на [LangGraph](https://www.langchain.com/langgraph), которая способна за **3-4** секунд проверять статистические гипотезы, по табличным данным.\
Используя проверки условий применения, система на подберет статистический тест, и проверит гипотезу. После, предоставит полноценную сводку исследования.

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

# Installation
1. Клонирование репозитория:
   git clone https://github.com/dmibar/Sirius_BC26.git
    cd Sirius_BC26
2. Создание виртуального окружения:\
Windows:
* python -m venv venv
    * venv\Scripts\activate


MacOS/Linux
* python3 -m venv venv
    * source venv/bin/activate

3. Установка зависимостей\
pip install -r requirements.txt\
pip install ipykernel\
python -m ipykernel install --user --name=venv --display-name "Python (Sirius_BC26)"

4. Запуск\
Откройте [code](https://github.com/dmibar/Sirius_BC26/blob/main/Code/CodeOfSystem.ipynb).ipynb через Jupyter (например VS Code)\
В правом верхнем углу (выбор Kernel) выберите созданное окружение: "Python (Sirius_BC26)".

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


# English
This repository contains code for a system on [LangGraph](https://www.langchain.com/langgraph), which is capable of testing statistical hypotheses using tabular data in **6-10** seconds.\
Using application condition checks, the LLM-based system will select a statistical test and test the hypothesis. Afterwards, it will provide a full summary of the study.
