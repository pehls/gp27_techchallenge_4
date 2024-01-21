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
tab_volatilidade, tab_crises_do_petroleo, tab_conflitos_armados, tab_energia_consumo, tab_exportacao = st.tabs(['Volatilidade', 'Crises do Petróleo', "Conflitos", 'Combustíveis Fósseis', 'Exportação de Comb.'])

# https://www.linkedin.com/pulse/petr%C3%B3leo-uma-an%C3%A1lise-sobre-volatilidade-yu64c/?originalSubdomain=pt
# https://pt.wikipedia.org/wiki/Crises_do_petr%C3%B3leo
# https://acleddata.com/data-export-tool/
# https://data.worldbank.org/topic/energy-and-mining - uso de energia, consumo de combustiveis fosseis, exportacao
with tab_volatilidade:
    st.markdown("""
        A idéia aqui é calcular a volatilidade do Preço do Petróleo, e a partir dos picos e vales sugerir como resposta os próximos itens nas abas
                
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
    subtab_uso_energia, subtab_consumo_comb_fosseis, subtab_correlacao_causalidade = st.tabs(['Uso de Energia', 'Consumo de Comb. Fósseis', 'Correlação e Causalidade'])
    
    with subtab_uso_energia:
        df_uso_energia_top10 = get_data._df_energy_use_top10()
        df_uso_energia, lista_2015_nulo = get_data._df_energy_use(list(df_uso_energia_top10['Country Name'].values))
        st.markdown(f"""
        Além da incidência de conflitos em relação ao Preço do Petróleo, vamos analisar a relação do uso de energia primária (combustível fóssil), antes da transformação para outros combustíveis de utilização final.
        O indicador aqui apresentado refere-se a produção interna mais importações e variações de estoque, menos as exportações e combustíveis fornecidos a navios e aeronaves utilizados no transporte internacional.
        O dado pode ser obtido em [Energy use (kg of oil equivalent per capita)](https://data.worldbank.org/indicator/EG.USE.PCAP.KG.OE);
                    
        Primeiramente, vamos analisar quais são os países com maior uso, nos 5 anos mais recentes de dados (entre 2011 e 2015):
        """)
        st.plotly_chart(
            generate_graphs._plot_top10_recent_energy_use(df_uso_energia_top10),
            use_container_width=True
        )
        st.markdown(f"""
        Por intuição, pontuaríamos os Estados Unidos como um dos 5 maiores, mas podemos perceber que ele pontua apenas em 10º lugar. Interessante ainda, ver Trinidade e Tobago e Curacao como membros do top5.
        
        Continuando, vamos verificar como esses Top 10 se relacionam com o Preço do Petróleo, observando um período anterior a 2015. Para tal, preenchemos o valor do indicador no ano de 2015, onde o mesmo era nulo, para os países:               
        """)
        
        for x in lista_2015_nulo:
            st.markdown(x)

        st.markdown(f"""
        Ainda, para facilitar a visualização de variáveis em diferentes unidades, os valores foram normalizados pela máxima e mínima, pontuando como 0 quando no mínimo, e 1 no máximo, considerando em separado o preço do Petróleo e os dados de uso de energia nos países;
        """)
        st.plotly_chart(
            generate_graphs._plot_energy_use(df_uso_energia),
            use_container_width=True
        )
        st.markdown(f"""
        Aqui, podemos destacar a proximidade dos preços do petróleo, e os valores do indicador nos países de Trinidad and Tobago e Brunei Darussalam entre os anos 1987-1997;
        entre os anos 1998/1999, o valor do preço do Petróleo aparenta ser inversamente proporcional ao uso de Curacao, por exemplo, que se amplifica fortemente;
        A partir de 1999, até 2008, vemos uma crescente bem forte do preço, praticamente em linha reta; da mesma forma, o indicador da Islândia (Iceland) se torna extremamente alto, se aproximando do valor máximo, que é admitido pelo Qatar em 2004; 
        A utilização de combustíveis fósseis tem um vale bem forte, juntamente com o seu preço, [no ano de 2009](https://www.politize.com.br/crise-financeira-de-2008/), período de uma crise mundial em decorrência da falência do banco de investimento norte-americano Lehman Brothers, que provocou uma [recessão econômica global](https://repositorio.ipea.gov.br/bitstream/11058/6248/1/RTM_v3_n1_Cinco.pdf), levando uma cadeia de falências de outras grandes financeiras.
        Em uma rápida recuperação, temos um novo pico, e o recorde para o período para o preço do petróleo em 2011, seguido por um aumento considerável da utilização do mesmo para países como Qatar, Curacao, Brunei Darussalam e os demais, seguidos por uma regularização do preço do petróleo em 2015, mas sem impactos significativos na utilização do mesmo para os países. 
        Neste ano de 2015, tendo iniciado em 2014 e findado em 2017, acontece uma nova crise econômica, com alguns fatores como o [fim do "superciclo das commodities"](https://brasil.elpais.com/brasil/2015/10/18/internacional/1445174932_674334.html), bem como uma desaceleração da economia chinesa, que vinha auxiliando a recuperação global desde 2008, e ainda uma ressaca econômica derivada do endividamento de muitos países europeus, [e até mesmo no Brasil](https://agenciabrasil.ebc.com.br/economia/noticia/2017-09/editada-embargo-10h-queda-de-2015-interrompeu-ascensao-do-setor-de-servicos).   
        """)

    with subtab_consumo_comb_fosseis:
        df_fuel_cons = get_data._df_fossil_fuel_cons()
        
        st.markdown(f"""
        Após dissertarmos sobre o uso de Energia Primária, vamos analisar o Consumo de Combustíveis Fósseis dos países, buscando identificar uma relação entre o uso per capita e o quanto os combustíveis fósseis representam, em percentual, da utilização derivada da matriz energética de cada país; 
        O dado pode ser obtido em [Fossil fuel energy consumption](https://data.worldbank.org/indicator/EG.USE.COMM.FO.ZS);
        """)
        st.write(df_fuel_cons)

        st.markdown("""
        Notamos aqui, que a maioria dos países está na região do meio-oeste e norte da África, países produtores de Petróleo; as exceções, Gibraltar e Brunei Darussalam, produzem e exportam Petróleo, em regiões diferentes.
        """)


    with subtab_correlacao_causalidade:
        df_fuel_corr_causa = get_data._get_fossil_fuel_cons_corr()
        st.markdown("""

        """)
    st.write(df_fuel_corr_causa)
    st.plotly_chart(
        generate_graphs._plot_correlation_matrix(
            get_data._events_correlations(df_fuel_corr_causa, cols_to_plot=['Fuel Consumption','Preco'])
            ), use_container_width=True
    )

with tab_exportacao:
    df_fuel_exp = get_data._df_fuel_exports()
    st.markdown("""
    Assim como o Consumo de Combustíveis fósseis, o dado da porcentagem da exportação de combustíveis, incluindo combustíveis minerais, lubrificantes e materiais relacionados, 
    está disponívei no world bank, através do [Fuel Exports (% of merchandise exports)](https://data.worldbank.org/indicator/TX.VAL.FUEL.ZS.UN?end=2022&start=2022&type=shaded&view=map&year=2022).
                
    As estimativas são feitas através da plataforma WITS da base de dados Comtrade mantida pela Divisão de Estatística das Nações Unidas.
    """)
    st.write(df_fuel_exp)

    st.markdown("""
    Assim como na utilização, temos 4 países do meio-oeste/norte da África (Libia, Kuwait, Qatar, Emirados Árabes Unidos), e mais 3 países da África SubSaariana (Angola, Nigeria, Rep. do Congo). Fechando o top 10, temos ainda Brunei Darussalam, Azerbaijão e Timor-Leste. 
    Tais países possuem mais de 67% de sua economia vinculada a combustíveis fósseis. sendo que o top 5 está muito próximo ou acima de 90% de sua economia vinculada ao Petróleo.
    """)
    