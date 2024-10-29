import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from yfinance import exceptions
import openai
import numpy as np
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests

api_key = st.secrets["API_KEY"]


def get_ticker_for_company(company_name):
    prompt = f"provide (ONLY!) the ticker for Yahoo Finance for the company, if user provides ticker use it: {company_name}"
    
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Only response, short as possible, no dot on end"},
            {"role": "user", "content": prompt}
        ]
    )
    
    ticker = response.choices[0].message.content.strip()
    return ticker


def summarize_recommendations(df):
    
    table_string = df.to_string(index=True)
    
    prompt = f"""Przeanalizuj poniższą tabelę, która przedstawia  procentowy udział poszczególnych rekomendacji w łącznej liczbie rekomendacji analityków na przestrzeni ostatnich miesięcy. Określ, czy widoczna jest tendencja wzrostowa lub spadkowa w liczbie rekomendacji dotyczących kupna i sprzedaży, z naciskiem na „Silne rekomendacje kupna” oraz „Silne rekomendacje sprzedaży” (te dwie kolumny mają kluczowe znaczenie). 
Skup się na najnowszych danych (0m) (o ile są dostępne), ale uwzględnij też wcześniejsze okresy: -1m to rekomendacje sprzed misiąca, -2m to sprzed dwóch a -3m to sprzed 3. Zamiast pisać -0m to pisz że aktualne rekomendacje, zamiast pisać -1m pisz że rekomendacje sprzed miesiąca, itp. Aby wychwycić ewentualne zmiany. Na końcu oceń, czy rekomendacje analityków są ogólnie pozytywne, neutralne czy negatywne dla zakupu akcji. Jeśli tabela nie zawiera danych, po prostu stwierdź, że brak jest rekomendacji analityków i nie wydawaj oceny czy jest to negatywne/pozytywne.
Napisz to krotko, 3-4 zdania. Najwazniejsze jest podsumowanie czy ogólnie oceny są Pozytywne/neutralne/negatywne dla inwestycji w akcje tej firmy. Napisz pogrubieniem któryś z wyrazów Pozytywne/neutralne/negatywne.

Poniższa tabela zawiera dane:
{df}
"""
    
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno że analitycy sugerują sprzedaż lub kupno akcji"},
            {"role": "user", "content": prompt}
        ]
    )

    summary = response.choices[0].message.content.strip()
    return summary

def summarize_indicators(df):

    table_string_indc = df.to_string(index=True)
    
    prompt = f"Przeanalizuj krótko sytuację finansową firmy na podstawie wskaźników:\n\n{table_string_indc}\n\n uwzględnij to w jakiej branży działa firma {company_name} wskaźniki z tej samej kategorii analizuj razem, w jednym punkcie (na przykład płynność) i dodawaj ocenę wskaźników przy każdej kategorii: Negatywne/Neutralne/Pozytywne pod kątem zakupu akcji, na końcu przy podsumowaniu pisz ocenę ogólną pogrubieniem(ale tylko słowa Negatywne/Neutralne/Pozytywne mają być pogrubione). Poszczególne kategorie oddzielaj linią. Bierz pod uwgę też aktualny poziom wskaźników danej kategorii, czy są na odpowiednim poziomie. Nie pisz nagłówka, ale podpunkty z pogrubieniem czego dotyczy pisz. Nie wypisuj wszystkich danych w tekście, użytkownik będzie miał tablkę załączoną. Pamiętaj też że najświeższe dane mogą być międzyokresowe i ewentualne odchylenia mogą wynikać z sezonowości danej firmy."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno czy dobre czy złe wskaźniki finansowe"},
            {"role": "user", "content": prompt}
        ]
    )

    summary_ind = response.choices[0].message.content.strip()
    return summary_ind

def summarize_market_indicators(df):

    indicators_2_df_str = df.to_string(index=True)
    
    prompt = f"Przeanalizuj krótko sytuację finansową firmy na podstawie wskaźników:\n\n{indicators_2_df_str}\n\n uwzględnij to w jakiej branży działa firma {company_name} wskaźniki z tej samej kategorii analizuj razem, w jednym punkcie (na przykład płynność) i dodawaj ocenę wskaźników przy każdej kategorii: Negatywne/Neutralne/Pozytywne pod kątem zakupu akcji, na końcu przy podsumowaniu pisz ocenę ogólną pogrubieniem(ale tylko słowa Negatywne/Neutralne/Pozytywne mają być pogrubione). Poszczególne kategorie oddzielaj linią. Bierz pod uwgę też aktualny poziom wskaźników danej kategorii, czy są na odpowiednim poziomie. Nie pisz nagłówka, ale podpunkty z pogrubieniem czego dotyczy pisz. Nie wypisuj wszystkich danych w tekście, użytkownik będzie miał tablkę załączoną. Pamiętaj też że najświeższe dane mogą być międzyokresowe i ewentualne odchylenia mogą wynikać z sezonowości danej firmy."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno czy dobre czy złe wskaźniki finansowe"},
            {"role": "user", "content": prompt}
        ]
    )

    summary_stock_ind = response.choices[0].message.content.strip()
    return summary_stock_ind


def summarize_financials_with_percent_changes(percent_changes, basic_fin, company_name):

    table_string_percent_changes = percent_changes.to_string(index=True)
    table_string_basic = basic_fin.to_string(index=True)
    
    prompt = f"Przeanalizuj krótko sytuację finansową firmy {company_name} na podstawie procentowych zmian danych finansowych i podstawowych danych finansowych:\n\nProcentowe zmiany danych finansowych:\n{table_string_percent_changes}\n\nPodstawowe dane finansowe:\n{table_string_basic}\n\n Uwzględnij to, w jakiej branży działa firma {company_name}. Dane z tej samej kategorii analizuj razem, w jednym punkcie i dodawaj ocenę grupy danych przy każdej kategorii: Negatywne/Neutralne/Pozytywne pod kątem zakupu akcji, Pisz to na końcu z pogrubieniem, na końcu przy podsumowaniu pisz ocenę ogólną pogrubieniem. Poszczególne kategorie oddzielaj linią. Bierz pod uwagę w jakiej branży działa firma, czy taki poziom poszczególnych danych jest charakterystyczny dla danego sektora. Nie pisz nagłówka, ale podpunkty z pogrubieniem czego dotyczy pisz. Nie wypisuj wszystkich danych w tekście, użytkownik będzie miał tabelę załączoną. Pamiętaj też, że najświeższe dane mogą być międzyokresowe i ewentualne odchylenia mogą wynikać z sezonowości danej firmy."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno czy dobre czy złe wskaźniki finansowe"},
            {"role": "user", "content": prompt}
        ]
    )

    summary_financials = response.choices[0].message.content.strip()
    return summary_financials


def Strenghts(company_name):

    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie mocne strony ma, staraj się pisać głównie te rzeczy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - najmocniejsze strony i krótko je rozwiń dlaczego akurat to jest ich mocną stroną na tle konkurencji."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Strenghts = response.choices[0].message.content.strip()
    return Strenghts


def Weaknesses(company_name):

    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie słabe strony ma, jakie słabości, staraj się pisać głównie te rzeczy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - najsłabsze strony i krótko je rozwiń dlaczego akurat to jest ich słabą stroną na tle konkurencji."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Weaknesses = response.choices[0].message.content.strip()
    return Weaknesses

def Opportunities(company_name):

    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie szanse przed nią stoją, staraj się pisać głównie te okazje dla firmy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - największe szanse i krótko je rozwiń dlaczego akurat to może być ich szansą na tle konkurencji."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Opportunities = response.choices[0].message.content.strip()
    return Opportunities


def Threats(company_name):

    prompt = f"Przeanalizuj krótko sytuację firmy {company_name}, napisz jakie zagrożenia przed nią stoją, staraj się pisać głównie o tych zagrożeniach dla firmy które wyróżniają ją na tle konkurencji. Nie pisz wstępu. Pisz 3 główne punkty - największe zagrożenia i krótko je rozwiń dlaczego akurat to może być ich zagrożeniem na tle konkurencji."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Threats = response.choices[0].message.content.strip()
    return Threats



def SWOT_summary(S,W,O,T,name_ticker):

    prompt = f"Napisz krótkie podsumowanie wszystkich składowych analizy SWOT firmy {name_ticker}. Mocnych stron: {S}, Słabych stron: {W}, Okazji: {O} i zagrożeń: {T}. Na końcu podsumowania napisz jedno zdanie w którym POGRUBIENIEM! napiszesz czy na podstawie analizy SWOT sytuacja jest Negatywna lub Neutralna lub pozytywna dla zakupu akcji tej firmy. Nie pisz podpunktów, napisz podsumowanie ciągiem. Ma mieć maksymalnie 5 zdań."

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    SWOT_summary = response.choices[0].message.content.strip()
    return SWOT_summary

def Report_summary(Recommendations, Basic_fin, Fin_ind, Market_ind, SWOT, NEWS):

    prompt = f"Napisz krótkie podsumowanie raportu na temat inwestycji w {company_name}, podsumowanie oprzyj na tych danych: {Recommendations}, {Basic_fin}, {Fin_ind}, {Market_ind}, {SWOT}, {NEWS}. Podsumowanie musi zawierać odniesienie do każdego fragmentu ale nie pisz tego samego, już nie podawaj masy liczb. Na koniec ma być zdanie oceniające czy aktualnie warto kupować czy nie. Staraj sie możliwie rzadko używać określenia że trudno określic. Rekomendację napisz pogrubieniem. Zawsze na koniec musi być czy według stanu na dzień dzisiejszy: Nie warto inwestować, Warto inwestować, Nie da się jednoznacznie określić. Nie pisz nagłówka. "

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    Report_summary = response.choices[0].message.content.strip()
    return Report_summary

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive'
}

def get_article_links(ticker):
    url = f'https://finance.yahoo.com/quote/{ticker}/latest-news/'
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()  # Sprawdza, czy kod statusu jest 200
    except requests.exceptions.RequestException as e:
        print(f"Nie udało się pobrać strony: {url}. Błąd: {e}")
        return []
    
    soup = BeautifulSoup(response.text, 'html.parser')
    article_elements = soup.select('a.subtle-link')
    
    article_links = set()  
    for element in article_elements:
        article_url = element.get('href')
        if article_url and article_url.startswith('/'):
            article_url = f"https://finance.yahoo.com{article_url}"
        
        if article_url.startswith('https://finance.yahoo.com/news/'):
            article_links.add(article_url)  
    
    return list(article_links)

def scrape_article_text(url):
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Nie udało się pobrać artykułu: {url}. Błąd: {e}")
        return "Brak treści."
    
    soup = BeautifulSoup(response.text, 'html.parser')
    article_text = soup.find('div', class_='body yf-5ef8bf')
    
    return article_text.get_text().strip() if article_text else "Brak treści."

def create_articles_dataframe(ticker):
    links = get_article_links(ticker)
    
    data = {
        "Link": [],
        "Treść": []
    }
    
    for link in links[:10]:  # Ograniczenie do 10 artykułów
        # Pomijanie linków, które prowadzą tylko do "https://finance.yahoo.com/news/"
        if link == "https://finance.yahoo.com/news/":
            continue
        
        text = scrape_article_text(link)
        data["Link"].append(link)
        data["Treść"].append(text)
    
    df = pd.DataFrame(data)
    return df

client = openai.OpenAI(api_key=api_key)

def streszczenie_artykułów(df):

    df['Streszczenie'] = ''
    
    for index, row in df.iterrows():
        content = row['Treść']
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Streść ten artykuł: {content}. Poszukaj w nim informacji na temat {company_name}. Oceń i napisz pogrubieniem czy w artykule pisano o {company_name} który był pozytywny dla inwestycji w akcje tej firmy, neutralny czy negatywny."}
            ]
        )
        
        summary = response.choices[0].message.content
        
        df.at[index, 'Streszczenie'] = summary
    
    return df

def summarize_news(df_streszczenia_to_AI):

    table_string_indc = df_streszczenia_to_AI.to_string(index=True)
    
    prompt = f"Przeanalizuj najnowsze wiadomości dotyczące {company_name} zawarte w tabeli:\n\n{table_string_indc}\n\n. Napisz podsumowanie wiadomości a w ostatnim zdaniu napisz pogrubieniem czy jest to negatywne lub neutralne lub pozytywne pod kątem inwestycji w akcje firmy {company_name}. Jeśli tabela z wiadomościami jest pusta to napisz tylko Brak danych"

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Ma to być fragment do raportu inwestycyjnego, same konkrety, pisz jasno."},
            {"role": "user", "content": prompt}
        ]
    )

    summary_news = response.choices[0].message.content.strip()
    return summary_news


def safe_calculate(func, *args):
    try:
        result = func(*args)
        if pd.isna(result):
            return None
        return result
    except (TypeError, ZeroDivisionError, KeyError):
        return None


st.title('Raport inwestycyjny')

company_name = st.text_input("Wprowadź nazwę firmy", "")

if st.button('Wygeneruj raport'):

    ticker = get_ticker_for_company(company_name)

    if not ticker:
        st.warning("Please enter a ticker")
    else:
        try:
            stock = yf.Ticker(ticker)
            company_name = stock.info.get('longName')

            if not company_name:
                st.error(f"Incorrect ticker: {ticker} or Yahoo Finance error")
            else:
                # Rekomendacje analityków
                recommendations = stock.recommendations
                name_ticker = f"{company_name} {ticker}"

                if recommendations is None or recommendations.empty:
                    st.warning("Brak rekomendacji analityków dla tej firmy.")
                else:
                    recommendations = recommendations.reset_index().set_index('period')
                    recommendations.drop(columns=['index'], inplace=True)
                    
                    recommendations = recommendations.loc[~(recommendations[['strongBuy', 'buy', 'hold', 'sell', 'strongSell']] == 0).all(axis=1)]
                    recommendations = pd.DataFrame(recommendations)
                    
                    st.subheader(f'Rekomendacje analityków dla {company_name} ({ticker})')

                    info = stock.info
                    currency_name = info.get('currency')

                    recommendations = recommendations.rename(columns={
                        'strongBuy': 'Silne rekom. kupna',
                        'buy': 'Rekom. kupna',
                        'hold': 'Rekom. trzymaj',
                        'sell': 'Rekom. sprzedaży',
                        'strongSell': 'Silne rekom. sprzedaży'
                    })
                    
                    total_recommendations = recommendations[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']].sum(axis=1)

                    recommendations_percentage = recommendations.copy()
                    recommendations_percentage[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']] = \
                        (recommendations[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']].div(total_recommendations, axis=0)) * 100

                    recommendations_percentage[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']] = \
                        recommendations_percentage[['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']].round(0).astype(int)
                    
                    st.markdown("Tabela poniżej przedstawia procentowy udział poszczególnych rekomendacji w łącznej liczbie rekomendacji:")
                    
                    for column in ['Silne rekom. kupna', 'Rekom. kupna', 'Rekom. trzymaj', 'Rekom. sprzedaży', 'Silne rekom. sprzedaży']:
                        recommendations_percentage[column] = recommendations_percentage[column].astype(str) + '%'
                    
                    st.table(recommendations_percentage)
                    
                    with st.expander("Rozwiń, aby zobaczyć tabelę z liczbą poszczególnych rekomendacji"):
                        st.dataframe(recommendations)
                
                summary = summarize_recommendations(recommendations_percentage)
                st.markdown(summary)

                # Dywidendy
                dividends = stock.dividends

                results = []

                for date, dividend in dividends.items():
                    history = stock.history(start=date, end=date + pd.Timedelta(days=1))
                    if not history.empty:
                        close_price = history['Close'].iloc[0]
                        dividend_yield = (dividend / close_price) * 100
                        results.append({
                            "Date": date,
                            "Dividend": dividend,
                            "Close Price": close_price,
                            "Dividend Yield (%)": dividend_yield
                        })

                dividend_df = pd.DataFrame(results)

                history_df = stock.history(period="max")[['Open', 'Close']].reset_index()
                history_df_close_only = history_df[['Date', 'Close']]
                history_df_close_only['Date'] = pd.to_datetime(history_df['Date']).dt.date
                history_df_close_only.set_index('Date', inplace=True)

                # Finanse roczne i kwartalne
                annual_financials = stock.financials.T.sort_index(ascending=False)
                annual_balance_sheet = stock.balance_sheet.T.sort_index(ascending=False)
                quarterly_financials = stock.quarterly_financials.T.sort_index(ascending=False)
                quarterly_balance_sheet = stock.quarterly_balance_sheet.T.sort_index(ascending=False)

                latest_annual_financial_date = annual_financials.index[0]
                latest_quarterly_financial_date = quarterly_financials.index[0]
                latest_annual_balance_date = annual_balance_sheet.index[0]
                latest_quarterly_balance_date = quarterly_balance_sheet.index[0]

                def normalize_to_annual(data, period_length_in_months):
                    return data * (12 / period_length_in_months)

                if latest_quarterly_financial_date > latest_annual_financial_date:
                    normalized_quarterly_financials = normalize_to_annual(quarterly_financials.head(1), 3)
                    latest_financials = pd.concat([normalized_quarterly_financials, annual_financials])
                else:
                    latest_financials = annual_financials

                if latest_quarterly_balance_date > latest_annual_balance_date:
                    latest_balance_sheet = pd.concat([quarterly_balance_sheet.head(1), annual_balance_sheet])
                else:
                    latest_balance_sheet = annual_balance_sheet

                latest_balance_sheet_2 = latest_balance_sheet
                latest_financials_2 = latest_financials

                # Funkcja do bezpiecznego obliczania
                def safe_calculate(func, *args):
                    try:
                        result = func(*args)
                        if pd.isna(result):
                            return None
                        return result
                    except (TypeError, ZeroDivisionError, KeyError):
                        return None

                results = []
                for date in latest_financials.index:
                    latest_financials_row = latest_financials.loc[date]
                    latest_balance_sheet_row = latest_balance_sheet.loc[date]
                    
                    current_assets = latest_balance_sheet_row.get('Current Assets')
                    current_liabilities = latest_balance_sheet_row.get('Current Liabilities')
                    inventory = latest_balance_sheet_row.get('Inventory')
                    total_liabilities = latest_balance_sheet_row.get('Total Liabilities Net Minority Interest')
                    total_equity = latest_balance_sheet_row.get('Total Equity Gross Minority Interest')
                    
                    # Obliczenia wskaźników z obsługą błędów
                    current_ratio = safe_calculate(lambda x, y: x / y, current_assets, current_liabilities)
                    quick_ratio = safe_calculate(lambda x, y, z: (x - y) / z, current_assets, inventory, current_liabilities)
                    debt_to_equity_ratio = safe_calculate(lambda x, y: x / y, total_liabilities, total_equity)
                    
                    total_revenue = latest_financials_row.get('Total Revenue')
                    cost_of_revenue = latest_financials_row.get('Cost Of Revenue')
                    gross_margin = safe_calculate(lambda x, y: x - y, total_revenue, cost_of_revenue)
                    gross_margin_ratio = safe_calculate(lambda x, y: (x / y) * 100, gross_margin, total_revenue)
                    
                    ebit = latest_financials_row.get('EBIT')
                    operating_margin = safe_calculate(lambda x, y: (x / y) * 100, ebit, total_revenue)
                    
                    net_income = latest_financials_row.get('Net Income')
                    net_profit_margin = safe_calculate(lambda x, y: (x / y) * 100, net_income, total_revenue)
                    
                    total_assets = latest_balance_sheet_row.get('Total Assets')
                    return_on_assets = safe_calculate(lambda x, y: (x / y) * 100, net_income, total_assets)
                    return_on_equity = safe_calculate(lambda x, y: (x / y) * 100, net_income, total_equity)
                    
                    results.append({
                        "Date": date.strftime("%Y-%m-%d"),
                        "Current Ratio": current_ratio,
                        "Quick Ratio": quick_ratio,
                        "Debt to Equity Ratio": debt_to_equity_ratio,
                        "Gross Margin Ratio (%)": gross_margin_ratio,
                        "Operating Margin (%)": operating_margin,
                        "Net Profit Margin (%)": net_profit_margin,
                        "Return on Assets (ROA) (%)": return_on_assets,
                        "Return on Equity (ROE) (%)": return_on_equity
                    })

                financial_metrics = ['Total Revenue', 'Total Revenue (Annualized)', 'EBIT', 'EBIT (Annualized)', 'Net Income', 'Net Income (Annualized)']
                balance_sheet_metrics = ['Total Assets', 'Total Liabilities Net Minority Interest', 'Total Equity Gross Minority Interest']

                data_fin = []

                for date in latest_financials.index:
                    row = {'Date': date.strftime("%d.%m.%Y")}
                    financials_row = latest_financials.loc[date]
                    balance_sheet_row = latest_balance_sheet.loc[date]
                    
                    for metric in financial_metrics:
                        if metric in financials_row:
                            row[metric] = financials_row[metric]
                    
                    for metric in balance_sheet_metrics:
                        if metric in balance_sheet_row:
                            row[metric] = balance_sheet_row[metric]
                    
                    data_fin.append(row)

                basic_fin = pd.DataFrame(data_fin)
                basic_fin = basic_fin.dropna(how='all')
                basic_fin = basic_fin.loc[~(basic_fin.isna().all(axis=1))]
                basic_fin.set_index('Date', inplace=True)

                column_mapping = {
                    'Total Revenue': 'Przychody Ogółem',
                    'EBIT': 'EBIT',
                    'Net Income': 'Zysk Netto',
                    'Total Assets': 'Aktywa Ogółem',
                    'Total Liabilities Net Minority Interest': 'Zobowiązania Ogółem',
                    'Total Equity Gross Minority Interest': 'Kapitał Własny',
                    'Total Revenue (Annualized)': 'Przychody Ogółem (Urocznione)',
                    'EBIT (Annualized)': 'EBIT (Urocznione)',
                    'Net Income (Annualized)': 'Zysk Netto (Uroczniony)'
                }

                basic_fin.rename(columns=column_mapping, inplace=True)

                def format_number(x):
                    if pd.isna(x):
                        return ''
                    elif isinstance(x, (int, float)):
                        return f'{x:,.0f}'.replace(',', ' ')
                    else:
                        return x

                basic_fin = basic_fin.map(format_number)

                basic_fin.dropna(how='all', subset=basic_fin.columns.difference(['Date']), inplace=True)
                basic_fin.fillna(0, inplace=True)
                basic_fin = basic_fin.loc[~((basic_fin == 0) | (basic_fin == '')).all(axis=1)]
                basic_fin.index = pd.to_datetime(basic_fin.index, format='%d.%m.%Y')

                basic_fin = basic_fin.sort_index(ascending=False)

                basic_fin_1 = basic_fin.apply(pd.to_numeric, errors='coerce')
                basic_fin.index = pd.to_datetime(basic_fin.index, format='%d.%m.%Y')
                basic_fin = basic_fin.sort_index(ascending=False)

                basic_fin_1 = basic_fin.replace(r'\s+', '', regex=True)
                basic_fin_1 = basic_fin_1.apply(pd.to_numeric, errors='coerce')

                percent_changes = (basic_fin_1 - basic_fin_1.shift(-1)) / basic_fin_1.shift(-1) * 100

                for col in percent_changes.columns:
                    previous_values = basic_fin_1[col].shift(-1)
                    current_values = basic_fin_1[col]
                    percent_changes[col] = np.where(
                        (previous_values < 0) & (current_values > 0),
                        (current_values - previous_values) / abs(previous_values) * 100,
                        percent_changes[col]
                    )

                percent_changes = percent_changes.round(0)
                percent_changes = percent_changes.dropna()
                percent_changes['Date'] = basic_fin.index.to_series().shift(-1).dt.strftime('%Y/%m') + " - " + basic_fin.index.to_series().dt.strftime('%Y/%m')
                percent_changes = percent_changes.set_index('Date')
                percent_changes = percent_changes.astype(str) + '%'
                percent_changes = pd.DataFrame(percent_changes)

                indicators_df = pd.DataFrame(results).set_index('Date')
                indicators_df.dropna(how='all', inplace=True)

                indicators_df = indicators_df.rename(columns={
                    'Date': 'Data',
                    'Current Ratio': 'Wskaźnik bież. płynności',
                    'Quick Ratio': 'Wskaźnik szybki',
                    'Debt to Equity Ratio': 'zadłużenie do kap. własnego',
                    'Gross Margin Ratio (%)': 'Marża brutto %',
                    'Operating Margin (%)': 'Marża operacyjna %',
                    'Net Profit Margin (%)': 'Marża zysku netto %',
                    'Return on Assets (ROA) (%)': 'Zwrot z aktywów (ROA) %',
                    'Return on Equity (ROE) (%)': 'Zwrot z kap. własnego (ROE) %'
                })
                
                summary_ind = summarize_indicators(indicators_df)
                summary_fin = summarize_financials_with_percent_changes(percent_changes, basic_fin, company_name)

                latest_financials_2.index = pd.to_datetime(latest_financials_2.index).date
                latest_balance_sheet_2.index = pd.to_datetime(latest_balance_sheet_2.index).date

                def get_closest_stock_price(date, history_df_close_only):
                    if date in history_df_close_only.index:
                        return history_df_close_only.at[date, 'Close']
                    
                    closest_date = min(history_df_close_only.index, key=lambda d: abs(d - date))
                    return history_df_close_only.at[closest_date, 'Close']

                indicators = []

                for date in latest_financials_2.index:
                    net_income = latest_financials_2.at[date, 'Net Income'] 
                    total_equity = latest_balance_sheet_2.at[date, 'Total Equity Gross Minority Interest']  
                    total_assets = latest_balance_sheet_2.at[date, 'Total Assets']
                    total_debt = latest_balance_sheet_2.at[date, 'Total Debt']
                    ebitda = latest_financials_2.at[date, 'EBITDA'] if 'EBITDA' in latest_financials_2.columns else 0

                    
                    shares = latest_balance_sheet_2.at[date, 'Share Issued']
                    
                    stock_price = get_closest_stock_price(date, history_df_close_only)
                    
                    market_cap = stock_price * shares if stock_price and shares else None

                    pe_ratio = safe_calculate(lambda x, y: x / y, market_cap, net_income)
                    book_value_per_share = safe_calculate(lambda x, y: x / y, total_equity, shares)
                    pb_ratio = safe_calculate(lambda x, y: x / y, market_cap, (book_value_per_share * shares)) if book_value_per_share else None
                    roe = safe_calculate(lambda x, y: x / y, net_income, total_equity)
                    roa = safe_calculate(lambda x, y: x / y, net_income, total_assets)
                    de_ratio = safe_calculate(lambda x, y: x / y, total_debt, total_equity)
                    ev = safe_calculate(
                        lambda x, y, z: x + y - z,
                        market_cap,
                        total_debt,
                        latest_balance_sheet_2.at[date, 'Cash Cash Equivalents And Short Term Investments'] if 'Cash Cash Equivalents And Short Term Investments' in latest_balance_sheet_2.columns else 0
                    ) if market_cap else None

                    ev_to_ebitda = safe_calculate(lambda x, y: x / y, ev, ebitda)

                    indicators.append({
                        'Date': date,
                        'Stock Price': stock_price,
                        'Market Cap': market_cap,
                        'P/E Ratio': pe_ratio,
                        'P/B Ratio': pb_ratio,
                        'ROE': roe,
                        'ROA': roa,
                        'Debt-to-Equity Ratio': de_ratio,
                        'EV/EBITDA': ev_to_ebitda,
                        'EPS': latest_financials_2.at[date, 'Basic EPS']  
                    })

                indicators_2_df = pd.DataFrame(indicators)
                indicators_2_df = indicators_2_df[['Date', 'P/E Ratio', 'P/B Ratio', 'ROE', 'ROA', 'EV/EBITDA', 'EPS']]
                indicators_2_df = indicators_2_df.dropna(how='all', subset=indicators_2_df.columns.difference(['Date']))

                AI_indicators_df = indicators_2_df.copy()
                AI_indicators_df['P/E Ratio'] = AI_indicators_df['P/E Ratio'].apply(lambda x: "Zysk poniżej zera" if x < 0 else x)
                AI_indicators_df['EV/EBITDA'] = AI_indicators_df['EV/EBITDA'].apply(lambda x: "Zysk poniżej zera" if x < 0 else x)

                indicators_2_df['P/E Ratio'] = indicators_2_df['P/E Ratio'].apply(lambda x: np.nan if isinstance(x, (int, float)) and x < 0 else x)
                indicators_2_df['EV/EBITDA'] = indicators_2_df['EV/EBITDA'].apply(lambda x: np.nan if isinstance(x, (int, float)) and x < 0 else x)
                indicators_2_df['Date'] = pd.to_datetime(indicators_2_df['Date'])
                indicators_2_df['Date'] = indicators_2_df['Date'].dt.strftime('%Y/%m')
                indicators_2_df.set_index('Date', inplace=True)
                indicators_2_df.sort_index(ascending=True, inplace=True)

                # Tworzenie wykresów
                fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 16))
                axes = axes.flatten()

                columns = ['P/E Ratio', 'P/B Ratio', 'ROE', 'ROA', 'EV/EBITDA', 'EPS']
                colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

                for ax, column, color in zip(axes, columns, colors):
                    bars = indicators_2_df[column].plot(kind='bar', ax=ax, color=color)
                    ax.set_title(column)
                    ax.set_xticklabels(indicators_2_df.index, rotation=45, ha='right')  
                    
                    for bar in bars.patches:
                        height = bar.get_height()
                        if not pd.isna(height):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                height,
                                f'{height:.2f}',
                                ha='center', va='bottom'
                            )

                plt.tight_layout()

                summary_stock_indc = summarize_market_indicators(AI_indicators_df)

                # Podstawowe dane finansowe
                st.subheader('Podstawowe dane finansowe')
                st.markdown(f"Poniższa tabela zawiera podstawowe dane finansowe {company_name}, dane międzyokresowe są urocznione w celu zapewnienia porównywalności do danych rocznych. Dane finansowe są przedstawione w walucie ***{currency_name}***")
                
                basic_fin.index = basic_fin.index.strftime('%Y/%m')
                st.dataframe(basic_fin)
                st.markdown(f"Zmiany między poszczególnymi okresami:")
                st.dataframe(percent_changes)
                st.markdown(summary_fin)
                
                # Analiza wskaźnikowa
                st.subheader(f'Analiza wskaźnikowa {company_name}')
                st.dataframe(indicators_df)
                st.markdown(summary_ind)

                # Podstawowe wskaźniki giełdowe
                st.subheader('Podstawowe wskaźniki giełdowe')
                st.pyplot(fig)
                st.markdown(summary_stock_indc)
                
                try:
                    if dividend_df.empty or history_df.empty:
                        st.warning("Brak danych o dywidendzie do wyświetlenia wykresów.")
                    else:
                        st.subheader('Dywidenda i stopa dywidendy na przestrzeni czasu')

                        fig1 = go.Figure()

                        fig1.add_trace(go.Scatter(
                            x=dividend_df['Date'], y=dividend_df['Dividend'],
                            mode='lines+markers', name='Kwota wypłaconej dywidendy',
                            line=dict(color='red'), marker=dict(color='red', size=4)
                        ))

                        fig1.add_trace(go.Scatter(
                            x=dividend_df['Date'], y=dividend_df['Dividend Yield (%)'],
                            mode='lines+markers', name='Stopa dywidendy (%)',
                            yaxis='y2', line=dict(color='blue'), marker=dict(color='blue', size=4)
                        ))

                        fig1.update_layout(
                            xaxis_title='Data',
                            yaxis=dict(title='Kwota wypłaconej dywidendy', showgrid=False),
                            yaxis2=dict(title='Stopa dywidendy (%)', overlaying='y', side='right'),
                            title='Dywidenda i stopa dywidendy na przestrzeni czasu',
                            legend=dict(x=0.01, y=1, orientation="h"),  
                            hovermode="x unified",
                            height=600  
                        )

                        fig1.update_xaxes(rangeslider_visible=True)
                        fig1.update_yaxes(autorange=True)
                        st.plotly_chart(fig1)

                        st.markdown("Powyższy wykres ilustruje zmiany w wysokości wypłacanej przez firmę dywidendy oraz stopę dywidendy w dniu jej wypłaty.")

                        st.subheader('Cena akcji i wypłacona dywidenda na przestrzeni czasu')

                        fig2 = go.Figure()

                        fig2.add_trace(go.Scatter(
                            x=history_df['Date'], y=history_df['Close'],
                            mode='lines', name='Cena zamknięcia akcji',
                            line=dict(color='green')
                        ))

                        fig2.add_trace(go.Scatter(
                            x=dividend_df['Date'], y=dividend_df['Dividend'],
                            mode='lines+markers', name='Kwota wypłaconej dywidendy',
                            yaxis='y2', line=dict(color='red'), marker=dict(color='red', size=4)
                        ))

                        fig2.update_layout(
                            xaxis_title='Data',
                            yaxis=dict(title='Cena zamknięcia akcji'),
                            yaxis2=dict(title='Kwota wypłaconej dywidendy', overlaying='y', side='right', showgrid=False),
                            title='Cena akcji i wypłacona dywidenda na przestrzeni czasu',
                            legend=dict(x=0.01, y=1, orientation="h"), 
                            hovermode="x unified",
                            height=600  
                        )

                        fig2.update_xaxes(rangeslider_visible=True)
                        fig2.update_yaxes(autorange=True)
                        st.plotly_chart(fig2)

                        st.markdown("Powyższy wykres ilustruje zmiany w wysokości wypłacanej przez firmę dywidendy oraz historyczne zmiany cen akcji.")

                except ValueError as e:
                    st.error(str(e))

                # Analiza SWOT
                st.subheader(f"Analiza SWOT firmy {company_name}:")        
                Strenghts_response = Strenghts(name_ticker)
                Weaknesses_response = Weaknesses(name_ticker)
                Opportunities_response = Opportunities(name_ticker)
                Threats_response = Threats(name_ticker)

                with st.expander("Wszystkie podpunkty analizy SWOT:"):

                    st.markdown("### Mocne strony:")
                    st.markdown(Strenghts_response)
                    
                    st.markdown("### Słabe strony:")
                    st.markdown(Weaknesses_response)
                
                    st.markdown("### Okazje:")
                    st.markdown(Opportunities_response)
                    
                    st.markdown("### Zagrożenia:")
                    st.markdown(Threats_response)

                SWOT_summary_response = SWOT_summary(Strenghts_response, Weaknesses_response, Opportunities_response, Threats_response, name_ticker)
                st.markdown(SWOT_summary_response)

                # Artykuły i wiadomości
                Articles = create_articles_dataframe(ticker)
                Articles = pd.DataFrame(Articles)
                Articles_short = streszczenie_artykułów(Articles)
                df_streszczenia_to_AI = Articles_short[['Streszczenie']]
                summarized_news = summarize_news(df_streszczenia_to_AI)
                News_links = Articles_short[['Link']]

                if not News_links.empty:  # Sprawdza, czy News_links nie jest pusty
                    st.subheader("Najnowsze wiadomości na temat firmy w pigułce:")
                    st.markdown(summarized_news)
                    with st.expander("Źródła wiadomości:"):
                        st.dataframe(News_links)

                # Podsumowanie raportu
                Report_summary_response = Report_summary(summary, summary_fin, summary_ind, summary_stock_indc, SWOT_summary_response, summarized_news)
                st.subheader(f"Podsumowanie raportu inwestycyjnego dotyczącego {company_name}")
                st.markdown(Report_summary_response)

        except Exception as e:
            st.error("Nieprawidłowy ticker")
            st.error(f"Error: {e}")
