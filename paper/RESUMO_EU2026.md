# Resumo — XLV Encontro de Iniciação Científica (Encontros Universitários UFC 2026)

> Rascunho pronto para colar no formulário de submissão
> (http://www.encontrosuniversitarios.ufc.br, período 22/06 a 13/07/2026) ou
> adaptar para resumo expandido. Preencha os campos entre `[...]` (autor,
> orientador, curso, unidade, ODS) antes de enviar. Lembre-se: o(a)
> orientador(a) precisa homologar o resumo na fase de ajustes
> (13/07–27/07/2026) para que a inscrição seja válida.

---

## Título

Redes Neurais Informadas por Física (PINNs) Aplicadas a Sistemas de
Reação-Difusão: Formação de Padrões de Turing no Sistema de Schnakenberg

## Autor(es)

[Nome completo do(a) autor(a) principal] — [Curso], [Unidade Acadêmica], UFC
Campus Quixadá

## Orientador(a)

[Nome do(a) orientador(a)] — [Departamento/Unidade]

## Palavras-chave

Redes Neurais Informadas por Física; Equações Diferenciais Parciais; Padrões
de Turing; Reação-Difusão; Aprendizado de Máquina Científico

## Área/Subárea sugerida

Ciência da Computação / Matemática Aplicada — Métodos Numéricos e
Computacionais

---

## Introdução

Equações diferenciais parciais (EDPs) não lineares descrevem grande parte
dos fenômenos de reação-difusão em física, biologia e química, mas sua
resolução analítica é, em geral, inviável, exigindo métodos numéricos como
diferenças finitas (FD), elementos finitos ou volumes finitos. Nos últimos
anos, as *Physics-Informed Neural Networks* (PINNs) — redes neurais
treinadas para satisfazer diretamente o resíduo da EDP via diferenciação
automática, além das condições iniciais e de contorno do problema — surgiram
como uma abordagem alternativa e livre de malha para a resolução de EDPs
(Raissi, Perdikaris & Karniadakis, 2019).

Este trabalho se insere em um projeto mais amplo de iniciação científica
que aplica PINNs a uma sequência de EDPs de complexidade crescente —
equação do calor (1D/2D), equação de Burgers (1D/2D) e, por fim, sistemas
de reação-difusão acoplados — com o objetivo de investigar a aplicabilidade
e as limitações desse método frente a soluções numéricas de referência já
validadas na literatura clássica de métodos de diferenças finitas.

O sistema escolhido para esta etapa é o **sistema de Schnakenberg**, um
modelo clássico de reação-difusão de duas espécies proposto para ilustrar a
**instabilidade de Turing**: um estado homogêneo estável na ausência de
difusão pode se tornar instável quando as taxas de difusão das duas
espécies diferem suficientemente, dando origem a padrões espaciais
espontâneos (manchas, listras) a partir de uma pequena perturbação. O
sistema de Schnakenberg é amplamente utilizado como *benchmark* padrão para
métodos numéricos aplicados a padrões de Turing, o que permite validação
direta contra soluções de referência bem estabelecidas.

## Objetivos

**Objetivo geral:** implementar e treinar uma PINN capaz de resolver o
sistema de reação-difusão de Schnakenberg em duas dimensões espaciais,
reproduzindo a formação de padrões de Turing a partir de uma perturbação do
estado homogêneo estacionário.

**Objetivos específicos:**
- Formular o resíduo da EDP do sistema de Schnakenberg (dois campos
  acoplados, condições de contorno de Neumann homogêneas) para treinamento
  via diferenciação automática;
- Construir uma solução numérica de referência (diferenças finitas /
  integração `BDF`) para validação quantitativa da PINN;
- Treinar a rede e analisar a evolução da função de perda (resíduo da EDP,
  condição inicial e condição de contorno) ao longo do treinamento;
- Identificar desafios específicos do treinamento de PINNs em sistemas
  rígidos (*stiff*) e com múltiplas escalas temporais, como o de
  Schnakenberg.

## Metodologia

A rede neural utilizada é um perceptron multicamadas totalmente conectado,
com entrada $(t, x, y)$, quatro camadas ocultas de 64 neurônios com ativação
$\tanh$, e saída $(u, v)$ correspondente aos dois campos do sistema. O
sistema de EDPs resolvido é:

$$
\frac{\partial u}{\partial t} = D_u \nabla^2 u + \gamma\left(a - u + u^2 v\right),
\qquad
\frac{\partial v}{\partial t} = D_v \nabla^2 v + \gamma\left(b - u^2 v\right)
$$

no domínio $\Omega = [0,1]\times[0,1]$, $t \in [0,10]$, com condição de
contorno de Neumann homogêneo (fluxo zero) nas quatro bordas e condição
inicial dada pelo estado homogêneo estacionário do sistema,
$(u^*, v^*) = \left(a+b,\; b/(a+b)^2\right)$, perturbado por ruído gaussiano
de pequena amplitude — sem essa perturbação, o sistema permaneceria
indefinidamente no estado homogêneo. Foi utilizado o conjunto clássico de
parâmetros de Murray para instabilidade de Turing ($a = 0{,}1305$,
$b = 0{,}7739$, $D_u = 1$, $D_v = 10$, $\gamma = 1000$), com razão
$D_v/D_u = 10$ suficiente para desencadear a instabilidade.

O treinamento minimiza uma função de perda composta,
$\mathcal{L} = \mathcal{L}_{PDE} + \mathcal{L}_{IC} + \mathcal{L}_{BC}$, cada
termo calculado via amostragem de pontos de colocação no domínio
espaço-temporal, avaliados com o otimizador Adam. As derivadas espaciais e
temporais necessárias para o resíduo da EDP e para as condições de contorno
de Neumann são obtidas por diferenciação automática (`torch.autograd`),
sem necessidade de discretização em malha.

Para validação, foi construída uma solução de referência independente por
integração numérica direta do sistema (método `BDF`, `scipy.integrate.
solve_ivp`), previamente verificada quanto à formação efetiva do padrão de
Turing esperado (aumento do desvio-padrão espacial do campo $u$ ao longo do
tempo) antes de ser adotada como referência de comparação.

## Resultados (parciais) e Discussão

O treinamento foi conduzido por 20.000 épocas em GPU (PyTorch, CUDA), com
acompanhamento periódico da função de perda. Ao final do treinamento, a
perda total convergiu para valores da ordem de $10^{-2}$-$10^{-3}$,
indicando que a rede satisfaz razoavelmente bem, em média, o resíduo da
EDP, a condição inicial e a condição de contorno nos pontos de colocação
amostrados.

A análise da curva de treinamento revelou um comportamento característico
de sistemas rígidos: oscilações e picos transitórios na componente de
resíduo da EDP em certas fases do treinamento, antes da estabilização.
Esse comportamento é consistente com desafios já documentados na literatura
de PINNs para problemas com múltiplas escalas — como o **viés espectral**
(tendência da rede a aprender primeiro componentes de baixa frequência) e
o desbalanceamento entre os termos da função de perda quando parâmetros
como $\gamma$ introduzem diferenças de escala acentuadas entre os termos de
reação e difusão do resíduo.

A comparação quantitativa e qualitativa entre a predição da PINN e a
solução de referência, bem como o ajuste fino da ponderação dos termos da
função de perda para lidar com a rigidez do sistema, constituem a etapa
atual e imediatamente seguinte do trabalho, e serão detalhados na
apresentação.

## Considerações Finais

Este trabalho evidencia tanto o potencial quanto os desafios concretos do
uso de PINNs para sistemas de reação-difusão não lineares e acoplados,
particularmente na presença de rigidez numérica. A investigação e correção
sistemática desses desafios de treinamento (reponderação dos termos de
perda, amostragem adaptativa de pontos de colocação, estratégias de
treinamento causal/curricular) representa a etapa metodológica central do
projeto, e o sistema de Schnakenberg foi deliberadamente escolhido como
piloto de baixo custo computacional para esse diagnóstico, antes de portar
as soluções encontradas para o sistema de difusão cruzada
predador-presa que constitui a etapa final do projeto de pesquisa.

## Referências

RAISSI, M.; PERDIKARIS, P.; KARNIADAKIS, G. E. Physics-informed neural
networks: A deep learning framework for solving forward and inverse
problems involving nonlinear partial differential equations. **Journal of
Computational Physics**, v. 378, p. 686–707, 2019.

TURING, A. M. The chemical basis of morphogenesis. **Philosophical
Transactions of the Royal Society of London. Series B**, v. 237, n. 641,
p. 37–72, 1952.

MURRAY, J. D. **Mathematical Biology II: Spatial Models and Biomedical
Applications**. 3. ed. New York: Springer, 2003.

[PEREIRA, R. R. **Métodos de Diferenças Finitas para Problemas de Difusão
e Reação Não Lineares**. Tese (Doutorado) — Laboratório Nacional de
Computação Científica (LNCC), 2019. — citar se o(a) orientador(a) confirmar
uso direto como referência de parâmetros/tabelas.]
