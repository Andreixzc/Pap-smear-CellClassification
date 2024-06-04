# BinaryXGB.py
Implementação do XGB partindo do pressuposto que as classes são binárias, (tem ou não a doença).
As caracteristicas usadas para treinar o modelo foram os momentos de hu previamente calculados e modificados para ter sua label binarizada, 'hu_moments_modified.csv'. 
O código treina o modelo e faz testas e armazena as métricas de desempenho, e também salva o modelo treinado em um arquivo PKL, para ser utilizado posteriormente.