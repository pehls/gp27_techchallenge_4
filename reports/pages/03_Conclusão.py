import streamlit as st

st.title('Conclusão')

st.markdown("""
    Com base nos dados obtidos, podemos verificar alguns pontos interessantes:
    - O uso de energia e o consumo de combustíveis fósseis de muitos países tem uma correlação forte ou muito forte com o preço do petróleo no decorrer dos anos, sendo a maioria delas positiva, o que indica que carregam o mesmo sentido, quando o preço está em alta, temos um alto consumo e uso, quando em baixa, os mesmos diminuem;
    - Alguns países, como Angola, tem uma forte dependência do petróleo, tendo em vista uma alta porcentagem de sua economia derivada da exportação, e um uso e consumo extremamente elevados;
    - Dentre as principais crises do petróleo, muitas delas são derivadas de conflitos armados, o que reflete em correlações elevadamente positivas na média dos valores comparando ao petróleo;
    - Na análise de feature importance destacamos 2: A primeira delas (e mais influente) a exportação/consumo de combustíveis fósseis, com um foco e importância maior na Índia (3º maior consumidor de petróleo no mundo), além de ter uma relação de causalidade de granger com o valor do mesmo;
    - A 2ª variável destacada e que reforça a interconexão entre os mercados financeiros e de commodities, é evidenciada pelo impacto influente do valor médio da Dow Jones na variação do preço do petróleo.
    Em suma, para o acompanhamento dos movimentos do preço de petróleo, devemos acompanhar movimentos de exportação, dada uma alta correlação da série temporal do petróleo com a exportação de alguns países, e o efeito de causalidade apresentado por alguns países, bem como o consumo de petróleo de países-chave, como Bolívia, Sudão, e alguns do norte da áfrica e oriente médio, uma região de forte incidência de conflitos e fatalidades, e o próprio índice Dow Jones, com um alto poder preditivo e uma correlação moderada com a série dos Preços.
            
    Ainda, para simular novos cenários, disponibilizamos na página de "Modelo" um modelo preditivo com um erro perto de 11%, com esta possibilidade. Podemos, portanto, imaginar cenários diferentes para cada índice ou informação apresentada (que possui uma certa série de dados para treinamento dos modelos), que ainda poderia ser trabalhado e melhorado para incluir diversas novas condições e testes em cima de diferentes variáveis e hipóteses.
""")
