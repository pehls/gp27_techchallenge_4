import streamlit as st
import config
from src import get_data, train_model, generate_graphs
from prophet.plot import plot_plotly, plot_components_plotly
from sklearn.metrics import (mean_absolute_error, 
                             mean_squared_error, 
                             mean_absolute_percentage_error,
                             r2_score)
from sklearn.utils import estimator_html_repr

st.title('Modelo')

tab_modelagem_inicial, tab_resultados_iniciais, tab_conceitos, tab_variaveis, tab_deploy_producao = st.tabs(['Modelagem', "Resultados", 'Conceitos', "Importância das Variáveis", "Plano - Deploy em Produção"])

df_petroleo = get_data._get_modelling_data()

with tab_modelagem_inicial:
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

    _model, X_test, pred, X_train, forecast_ = train_model._train_simple_prophet(df_petroleo)
    baseline_mape = round(mean_absolute_percentage_error(X_test['y'].values, pred['yhat'].values)*100, 3)
    baseline_r2 = round(r2_score(X_train['y'].values, forecast_['yhat'].values), 4)

with tab_resultados_iniciais:
    st.plotly_chart(
        plot_plotly(_model, forecast_.dropna()),
        use_container_width=True,
    )

    st.markdown(f"""
        De acordo com o gráfico acima, podemos ver que a previsão do modelo, embora com resultados interessantes,
        ainda carece de um ajuste melhor. 
                
        No período de teste, datado entre {min(pred.ds).date()} e {max(pred.ds).date()}, temos um erro médio absoluto percentual de 
        **{baseline_mape}%**,
        e um R2 (medida de ajuste na etapa de treinamento) de 
        **{baseline_r2}**
                
    """)

with tab_conceitos:
    st.markdown(f"""
        A partir dessa modelagem inicial, e da descoberta de uma sazonalidade de 365 dias, iremos utilizar alguns 
        conceitos mais avançados para melhorar o desempenho de nossa previsão:
                
        #### Validação Cruzada com Time Series Split
                
        Esta é uma técnica usada para avaliar o desempenho de modelos de aprendizado de máquina em dados de séries temporais. 
                
        A diferença para dessa técnica para a validação cruzada tradicional, onde os dados são embaralhados aleatoriamente, na validação cruzada com divisão de séries temporais, a ordem temporal dos dados é mantida, pois a dependência temporal é crucial em séries temporais.

        Sobre o processo:

        1. O 1º passo envolve a divisão do conjunto de dados em vários blocos ou dobras, onde cada bloco subsequente contém observações temporais mais recentes;
        2. Após isso, treinamos o modelo em cada conjunto de treinamento (treino)
        3. Na sequência avaliamos seu desempenho no conjunto de teste correspondente (teste). A métrica de avaliação é registrada para cada dobra.

    """)
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(config.TIME_SERIES_SPLIT_IMG,
                caption="Time Series Split",
                width=300,  
        )

    st.markdown(f"""     
        Com isso, teremos várias métricas de desempenho para cada dobra. Isso nos ajudará a entender como o modelo se comporta em diferentes períodos da série temporal 
        e se ele consegue generalizar bem para dados futuros.
                

        #### Adição de feriados na modelagem do Prophet
                
        A adição de feriados na modelagem do Prophet (que já possui um mecanismo embutido para lidar esses eventos sazonais, como, por exemplo, os feriados) é uma técnica usada para melhorar a precisão das previsões em séries temporais ao levar em consideração esses eventos.

        Essa implementação permitirá o modelo ajustar seus componentes de sazonalidade e tendência conforme os padrões observados nos dados de feriados passados, resultando em previsões mais precisas em relação a esses eventos especiais.
    """)
    
    st.markdown(f"""            
        #### Hiperparametrização Bayesiana
                
        É uma abordagem para encontrar os melhores hiperparâmetros ( configurações que não são aprendidas diretamente pelo algoritmo durante o treinamento, 
        mas afetam como o modelo é treinado e como faz previsões) para um modelo de machine learning.

        Ela utiliza modelos probabilísticos e estatísticos para prever como diferentes configurações afetarão o desempenho do modelo.
        Em vez de tentar todas as combinações possíveis (o que pode ser computacionalmente caro), ela usa um modelo substituto para guiar a busca, 
        focando nas combinações mais promissoras e equilibrando a exploração do que não conhece com o aproveitamento do que já foi aprendido.        
    """)

with tab_variaveis:
    df_final = get_data._df_tree_modelling()
    dict_results = train_model._run_xgboost(df_final)
    df_importances = train_model._get_tree_importances(dict_results['pipeline'])
    st.markdown("""
    Para a análise de quais features mais importam, treinaremos um segundo modelo - chamado XGBoost, conforme explicado nos conceitos.
    Abaixo, vemos os passos do pipeline de previsão:
    """)
    with open("D:/Cursos/FIAP_pós/gp27_techchallenge_4/models/pipeline.html", "r", encoding='utf-8') as f:
        html_pipe = f.read()
    st.write(
        html_pipe, unsafe_allow_html=True
    )
    st.write(html_pipe, unsafe_allow_html=True)
    st.markdown(f"Para esse modelo, o mape ficou em {dict_results['mape']}")

    st.plotly_chart(
        generate_graphs._plot_df_importances(df_importances),
        use_container_width=True
    )
with tab_deploy_producao:
    st.markdown(f"""
  
                
    """)