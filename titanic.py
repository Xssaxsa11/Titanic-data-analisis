import pandas as pd
from pathlib import Path



def clean_titanic_data(df):
    """Предобработка сырых данных."""
    df_clean = df.copy()
    # Удаляем ненужное
    df_clean = df_clean.drop(columns=['PassengerId', 'Ticket'], errors='ignore')
    # Заполняем пропуски в возрасте медианой
    df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
    return df_clean


def survival_rate(df):
    """
    Анализируем выжимаемость людей на корабле в зависимости от пола и возраста

    Args: 
        df (DataFrame) - очищенный датасет Титаника

    Returns:
        Два обьекта Series: (выживаемость по полу, выживаемость по возрасту)    
    """
    #Анализ выживаемости по полу
    sex_survival_rate = df.groupby('Sex')['Survived'].mean()
    sex_survival_rate = sex_survival_rate.apply(lambda x: str(round(x*100, 1)) + '%') #Возвращает таблицу с выживаемостью мужчин и женщин


    age_bind = [0, 12, 18, 60, 80]
    age_labels = ['Child', 'Teenager', 'Adult', 'Senior'] #создаю массивы для cut

    age_tabl = pd.cut(df['Age'], bins=age_bind, labels=age_labels)
    test_df = df.copy()
    age_idx = test_df.columns.get_loc('Age') #элемент для insert 

    

    test_df.insert(loc = age_idx + 1, column='Age_Group', value=age_tabl)
    
    ages_live_rate = test_df.groupby('Age_Group')['Survived'].mean()*100
    ages_live_rate = ages_live_rate.apply(lambda x: str(round(x, 1)) + '%')
    
    return sex_survival_rate, ages_live_rate


def wealth_survival_rate(df):
    """
    Анализируем выжимаемость людей на корабле в зависимости от класса, а также медианную стоимость билета по выжившим/умершим

    Args: 
        df (DataFrame) - очищенный датасет Титаника

    Returns:
        Два обьекта Series: (выживаемость по классу, медианная цена билета выживших/мертвых)    
    """

    survivors_per_pclass = df.groupby('Pclass')['Survived'].mean() #считаем процент выживших по классу

    df1 = df.copy()
    df1['Passenger_status'] = df1['Survived'].replace({0:'Dead', 1: 'Alive'})
    median_fare = df1.groupby('Passenger_status')['Fare'].median()
    
    return survivors_per_pclass, median_fare


def chech_families_survive_rate(df):
    """
    Анализируем выжимаемость людей на корабле в зависимости от количества людей в семье,

    Args: 
        df (DataFrame) - очищенный датасет Титаника

    Returns:
        Jбьект Series: выживаемость по количеству людей в семье    
    """

    family_bins = [-1, 0, 3, 100]
    family_labels = ['Single', 'Small_family', 'Big_family']
    amount_of_family = df['SibSp'] + df['Parch']

    df1 = df.copy()
    df1['Size_of_family'] = pd.cut(amount_of_family, bins=family_bins, labels=family_labels)

    family_survive_rate = df1.groupby('Size_of_family')['Survived'].mean()

    return family_survive_rate







if __name__ == '__main__':
    # Python сам поймет, что мы запустили titanic.py, и найдет его папку
    BASE_DIR = Path(__file__).resolve().parent
    
    # И будет искать csv-файл в этой же папке
    csv_path = BASE_DIR / 'titanic.csv'
    
    data = pd.read_csv(csv_path)
    df = clean_titanic_data(data)
    sex_survival_rate , age_rate = survival_rate (df)

    print("\n" + "="*60)
    print("      РЕЗУЛЬТАТЫ АНАЛИЗА ДАННЫХ ПАССАЖИРОВ ТИТАНИКА      ")
    print("="*60 + "\n")
    print("ГИПОТЕЗА №1: Выживали ли мужчины чаще чем женщины?")
    print("-" * 55)
    print(f'Выживаемость мужчин: {sex_survival_rate['male']}\nВыживаемость женщин: {sex_survival_rate['female']}')
    print(f"• Дети (до 12 лет):          {age_rate.loc['Child']}")
    print(f"• Подростки (12-18 лет):     {age_rate.loc['Teenager']}")
    print(f"• Взрослые (18-60 лет):      {age_rate.loc['Adult']}")
    print(f"• Пожилые (от 60 лет):       {age_rate.loc['Senior']}")
    print("\n" + "."*55 + "\n")

    class_rate, fare_rate = wealth_survival_rate(df)
    print("ГИПОТЕЗА №2: Влияние социально-экономического статуса")
    print("-" * 55)
    print(f"• Выживаемость в 1 классе (Люкс):   {class_rate.loc[1]:.1%}")
    print(f"• Выживаемость во 2 классе (Комфорт): {class_rate.loc[2]:.1%}")
    print(f"• Выживаемость в 3 классе (Эконом):  {class_rate.loc[3]:.1%}")
    print()
    print(f"• Медианная цена билета ПОГИБШИХ:    {fare_rate.loc['Dead']:.2f} £")
    print(f"• Медианная цена билета ВЫЖИВШИХ:    {fare_rate.loc['Alive']:.2f} £")
    print("\n" + "."*55 + "\n")

    family_rate = chech_families_survive_rate(df)
    print("ГИПОТЕЗА №3: Влияние размера семьи на спасение")
    print("-" * 55)
    print(f"• Одинокие пассажиры (Single):      {family_rate.loc['Single']:.1%}")
    print(f"• Небольшие семьи (1-3 родственника): {family_rate.loc['Small_family']:.1%}")
    print(f"• Большие семьи (более 3 человек):  {family_rate.loc['Big_family']:.1%}")
        
    print("\n" + "="*60)
    print("                    АНАЛИЗ ЗАВЕРШЕН                     ")
    print("="*60 + "\n")