import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import src.generate_graphs as generate_graphs
import src.get_data as get_data

st.write("""
    # Tech Challenge #02 - Grupo 27 
    ## Modelo Preditivo / IBOVESPA
    by. Eduardo Gomes, Igor Brito e Gabriel Pehls
""")
         
st.info("""
    Com objetivo de predizer a tendência de fechamento do IBOVESPA, mostramos nesse trabalho
    todo o processo para criação do nosso modelo.
    
    Os dados aqui utilizados foram baixados do site [investing](https://br.investing.com/indices/bovespa-historical-data) 
    e contemplam o período de 01-01-2020 até 31-07-2023.
""")
        

tab_grafico_historico, tab_seasonal, tab_adf, tab_acf, tab_models = st.tabs(['Gráfico Histórico', 'Decompondo sazonalidade', 'Teste ADFuller', 'Autocorrelação - ACF/PACF', 'Modelos - Teste'])

df_ibovespa = get_data._df_ibovespa()

with tab_grafico_historico:
    df, crossover, _, _, _, _, _, _, _, _ = get_data._get_all_indicators_data()
    st.plotly_chart(
        generate_graphs._grafico_historico(df, crossover),
        use_container_width=True,
    )

with tab_seasonal:
    st.markdown("""
        Utilizando a função `seasonal_decompose` não foi identificado nenhum padrão sazonal.
        Foi utilizado o valor 5 no parâmetro _period_ por ser esse o ciclo de dias da bolsa
    """)
    st.plotly_chart(
        generate_graphs._seasonal_decompose(get_data._series_for_seasonal()),
        use_container_width=True,
    )

    st.markdown("""
        Ao tratarmos o periodo como 365, porém, notamos um comportamento sazonal mais evidente:
    """)
    st.plotly_chart(
        generate_graphs._seasonal_decompose(get_data._series_for_seasonal(), 365),
        use_container_width=True,
    )
    
    st.markdown("""
        Seria um sinal de que temos um comportamento cíclico anual, e uma tendência bem mais definida?
                Nota-se que o gráfico de tendência está muito mais constante e conciso, com uma alta evidente até out/2022!
    """)
with tab_adf:
    grafico_adf, series_adf = generate_graphs._adf(df_ibovespa)
    res_adf = get_data._adfuller(series_adf)
    st.plotly_chart(
        grafico_adf,
        use_container_width=True,
    )

    st.markdown(f"""
        Aplicando a função `adfuller` ao dados sem nenhum tratamento, verificamos que a série não é estacionária
        ```
        Teste estatístico: {res_adf[0]}
        p-value: {res_adf[1]}
        Valores críticos: {res_adf[4]}
        ```
    """)

    st.divider()

    grafico_adf_diff, series_adf_diff = generate_graphs._adf_diff(df_ibovespa)
    res_adf_diff = get_data._adfuller(series_adf_diff)
    st.plotly_chart(
        grafico_adf_diff,
        use_container_width=True,                                 
    )

    st.markdown(f"""
        Normalizando os dados com a diferenciação conseguimos transformar a série em estacionária.
        ```
        Teste estatístico: {res_adf_diff[0]}
        p-value: {res_adf_diff[1]}
        Valores críticos: {res_adf_diff[4]}
        ```
    """)

with tab_acf:
    st.pyplot(
        plot_acf(df_ibovespa['Close'].values),
        use_container_width=True,
    )

    st.divider()

    st.pyplot(
        plot_pacf(df_ibovespa['Close'].values),
        use_container_width=True,
    )

with tab_models:
    st.markdown("""
        Antes de partirmos para o algoritmo selecionado para gerar o modelo, realizamos alguns testes com algoritmos
        para séries temporais
        
        Os algoritmos da lib `statsforecast` são:
        - SeasonalNaive
        - SeasonalWindowAverage
        - AutoARIMA
    """)

    graph, mape_seas_naive, mape_seas_wa, mape_arima = generate_graphs._models_ts()
    st.plotly_chart(
        graph,
        use_container_width=True,
    )

    st.markdown(f"""
        Para avaliação utilizamos `sklearn.metrics.mean_average_percentage_error` e os resultados são listados abaixo:
        - MAPE SeasonalNaive: {mape_seas_naive}
        - MAPE SeasonalWindowAverage: {mape_seas_wa}
        - MAPE AutoARIMA: {mape_arima}

        Na página [Modelo](Modelo) aprofundamos mais a análise do algoritmo `prophet`.
    """)
