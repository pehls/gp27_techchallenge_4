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
    subtab_conflitos_paises, subtab_tipo_evento_petroleo, subtab_correlacao_causalidade = st.tabs(['Países em Conflitos', 'Tipos de evento e Petróleo', 'Correlação e Causalidade'])

    with subtab_conflitos_paises:
        st.markdown("""
        Os dados foram obtidos no portal [Armed Conflict Location & Event Data Project](https://acleddata.com/data-export-tool/), um portal que coleta, analista e mapeia crises globais, 
        salvando informacoes diversas sobre tais conflitos em diferentes locais.
                    
        Para começar, analisemos os países com mais fatalidades nos anos analisados:
        """)

        st.plotly_chart(
            generate_graphs._plot_conflitos_paises(df_conflitos_porpais),
            use_container_width=True,
        )
        st.markdown("""
        Será que são os mesmos países com maior exportação de petróleo no períiodo? Será que existe alguma correlação entre as fatalidades e o preço do Petróleo?
                    
        Para respoder tais perguntas, precisamos entender os tipos de eventos que a plataforma entrega:
                    - Batalhas: Basicamente, são confrontos armados, com atores estatais ou não, sejam governos ou grupos armados apenas;
                    - Violência contra civis: são ataques, desaparecimentos forçados e sequestros e violências sexuais;
                    - Explosões / Violência Remoda: aqui estão incluídos ataques aéreos, com mísseis e artilharias, explosões remotas, ataques com drones ou por via aérea (aviões, por exemplo), granadas e bombas suicidas.
        
        Notamos, por essas descrições, que são eventos precisamente violentos, muitos derivados de confrontos geopolíticos, e até mesmo derivados de protestos com maior incidência de violência, como o que vemos atualmente na Ucrânia. 
                    Por tal evento ter um cunho territorial específico entre a Ucrânia e Rússia, com o anexo da Criméia, uma região que produz petróleo e gás, por exemplo, ela entra na categoria de Batalhas, e poderia ter uma forte relação com o aumento do petróleo na região e no mundo, a depender da força da produção do mesmo; Sabemos que existem outros motivos que poderiam determinar tal confronto, mas vamos focar nos impactos no preço do petróleo e sua produção mundial;
        """)

    with subtab_tipo_evento_petroleo:
        df_conflitos_preco_normalizados = get_data._events_normalized_globally()
        st.plotly_chart(
            generate_graphs._plot_conflitos_tipo(df_conflitos_porpais),
            use_container_width=True,
        )
        st.markdown("""
        De forma geral, notamos que a quantidade de fatalidades costuma ser maior com explosões e eventos relacionados, seguido por batalhas. Em poucos meses, e em menor quantidade de fatalidades, temos os casos de violência contra civis. Vamos analisar a quantidade de eventos e fatalidades vs Preço do Petróleo Brunt:
        """)
        st.plotly_chart(
            generate_graphs._plot_conflitos_tipo_e_petroleo(df_conflitos_preco_normalizados), use_container_width=True
        )
        st.markdown("""
        Visualmente, ao compararmos apenas o Preço vs quantidade total de eventos e de fatalidades, vemos poucos pontos relacionados, mas uma coerência bem forte entre os números no período de 2021 (começo do ano) e de 2022 (começo do ano), fato que não se repete em 2023 (notamos aqui, um aumento no número de eventos e fatalidades, mas o preço do petróleo não tem picos e vales tão bem estabelecidos). No decorrer do ano de 23, os meses de Março, Abril e Maio tem caracteristicas muito semelhantes nos três indicadores, com o preço em uma crescente no restante do ano, até o fim dos dados.
                    
        Outro ponto interessante é o número de explosões aparentar estar altamente relacionado com o Preço, mas será que o mesmo se repete quando analisado de forma estatística? Ou o comportamento visual apenas? Vamos analisar:
        """)
    with subtab_correlacao_causalidade:
        st.markdown("""
        Para analisar as relações estatisticamente, vamos usar dois tipos de correlação:
                    
        - **Correlação de Pearson**: talbém chamada de correlação produto-momento, mede o grau de correlação entre duas variáveis de escala métrica, no instante x, ou seja, de forma direta no mesmo ponto do tempo. Quão mais próxima de 1, seja negativo ou positivo, mais forte é, de acordo com o seu sentido. Caso muito próxima de 0, nota-se uma correlação fraca.
                    
        - **DTW**: O Dynamic Time Warping é um algoritmo utilizado para comparar e alinhar duas séries temporais, utilizando um alinhamento não-sincronizado da série, ou seja, as séries poderiam ter uma relação não exatamente no mesmo ponto do tempo. Quanto mais próximo de zero, maior a similaridade das duas séries temporais.
        
        E, ainda, um teste de causalidade:
                    
        - **Teste de Causalidade de Granger**: É um teste estatístico que visa superar as limitações do uso de simples correlações entre variáveis, analisando o sentido causal entre elas, demonstrando que uma variável X "Granger Causa" Y caso os valores do passado de X ajudam a prever o valor presente de Y. Tipicamente, aplica-se um teste F sobre os erros da predição de Y pelo seu passado, e da predição de Y pelo seu passado e pela variável X, visando testar a hipótese de que Y é predito apenas por seus valores passados (H0) e se o passado de X ajuda a prever Y (H1);

                    
        """)
        
        st.plotly_chart(
            generate_graphs._plot_correlation_matrix(
                get_data._events_correlations(df_conflitos_preco_normalizados)
                ), use_container_width=True
        )
        st.markdown("""
        # **PRECISO AUMENTAR ESSE GRAFICO**
        """)
        # ref https://www.questionpro.com/blog/pt-br/correlacao-de-pearson/
        # ref https://community.revelo.com.br/primeiros-passos-no-dtw/
        # ref https://www.linkedin.com/pulse/causalidade-de-granger-guilherme-garcia/?originalSubdomain=pt

with tab_crises_do_petroleo:
    st.image(config.PETR4_HISTORIC,
                caption="Crises do Petróleo",
                width=600,
        )
    st.markdown("""
    """)

with tab_energia_consumo:
    st.markdown("""
    """)

with tab_exportacao:
    st.markdown("""
    """)
    