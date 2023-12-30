import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)

st.title('Análise Histórica')

df_petroleo = get_data._df_petroleo()
df_conflitos_porpais = get_data._events_per_country()
df_conflitos_mundiais = get_data._events_globally()
tab_volatilidade, tab_crises_do_petroleo, tab_conflitos_armados, tab_energia_consumo, tab_exportacao = st.tabs(['A Volatilidade do Petróleo', 'Crises do Petróleo', "Conflitos Armados", 'Uso de Energia e Consumo de Comb. Fósseis', 'Exportação de Combustíveis'])

# https://www.linkedin.com/pulse/petr%C3%B3leo-uma-an%C3%A1lise-sobre-volatilidade-yu64c/?originalSubdomain=pt
# https://pt.wikipedia.org/wiki/Crises_do_petr%C3%B3leo
# https://acleddata.com/data-export-tool/
# https://data.worldbank.org/topic/energy-and-mining - uso de energia, consumo de combustiveis fosseis, exportacao
with tab_volatilidade:
    st.markdown("""
        Iniciando a etapa de modelagem, optamos por experimentar o modelo Prophet, criado pela Meta/Facebook em 2017, 
        sendo um algoritmo de previsão de séries temporais que lida de forma eficiente com séries onde temos uma presença 
        forte e destacada de Sazonalidades, Feriados previamente conhecidoss e uma tendência de crescimento destacada.
        O mesmo define a série temporal tendo três componentes principais, tendência (g), sazonalidade (s) e feriados (h), 
        combinados na seguinte equação:

        `y(t)=g(t)+s(t)+h(t)+εt`, onde εt é o termo de erro, basicamente.
        
        Iniciaremos com um modelo simples, para verificar seu desempenho, e partiremos para conceitos mais elaborados, 
        chegando a um modelo hiperparametrizado, e com um desempenho superior para a aplicação.
                
    """)

    st.plotly_chart(
        generate_graphs._grafico_historico(df_petroleo),
        use_container_width=True,
    )

with tab_conflitos_armados:
    st.markdown("""
    Os dados foram obtidos no portal [Armed Conflict Location & Event Data Project](https://acleddata.com/data-export-tool/), um portal que coleta, analista e mapeia crises globais, salvando informacoes diversas sobre tais conflitos em diferentes locais.
    """)

    st.plotly_chart(
        generate_graphs._plot_conflitos_paises(df_conflitos_porpais),
        use_container_width=True,
    )

    st.plotly_chart(
        generate_graphs._plot_conflitos_tipo(df_conflitos_porpais),
        use_container_width=True,
    )

with tab_crises_do_petroleo:
    st.markdown("""
    """)

with tab_energia_consumo:
    st.markdown("""
    """)

with tab_exportacao:
    st.markdown("""
    """)