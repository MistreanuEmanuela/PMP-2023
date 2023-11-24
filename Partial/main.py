import numpy as np

p1 = 0.5
p_n = 0.5
j_0_b =0
j_0_s =0
j_1_b =0
j_1_s = 0
p_j_1_s = 0.66

if np.random.rand() < p1:
    if np.random.rand() < p_n:
        j_0_s= j_0_s +1
    else:
        j_0_b = j_0_b +1
    for i in range(j_0_s + 1):
        if np.random.rand() < p_j_1_s:
            j_1_s = j_1_s + 1
        else:
            j_1_b = j_1_b + 1
    if(j_0_s > j_1_s):
        print("a castigat jucator 1")
else:
    if np.random.rand() <p_j_1_s:
        j_1_s= j_1_s +1
    else:
        j_1_b= j_1_b +1

    for i in range(j_1_s + 1):
        if np.random.rand() < p_n:
            j_0_s = j_0_s + 1
        else:
            j_0_b = j_0_b + 1
    if(j_0_s > j_1_s):
        print("a castigat jucator 1")


#la modul general:
castigatori = []
for i in range(1000):
    p1 = 0.5
    p_n = 0.5
    j_0_b = 0
    j_0_s = 0
    j_1_b = 0
    j_1_s = 0
    p_j_1_s = 0.66
#probabilitatile pentru evenimenti si numarul de steme, ban dat de fiecare jucator
#alegem random un nr intre 0 si 1, daca e mai mare ca 0.5 da jucator 1, daca nu da jucator0
    if np.random.rand() < p1:
        if np.random.rand() < p_n:
            j_0_s = j_0_s + 1
        else:
            j_0_b = j_0_b + 1
        for i in range(j_0_s + 1):
            if np.random.rand() < p_j_1_s:
                j_1_s = j_1_s + 1
            else:
                j_1_b = j_1_b + 1
        if (j_0_s >= j_1_s):
            castigatori.append(1) #mentinem intr un vector cine a castigat
        else:
            castigatori.append(2)
    else: #da j1 unu primul
        if np.random.rand() < p_j_1_s: #folosim faptul ca da cu prob 0.66 stema
            j_1_s = j_1_s + 1
        else:
            j_1_b = j_1_b + 1

        for i in range(j_1_s + 1):
            if np.random.rand() < p_n: #vedem ce da jucator 2
                j_0_s = j_0_s + 1
            else:
                j_0_b = j_0_b + 1
        if (j_0_s >= j_1_s):
            castigatori.append(1)
        else:
            castigatori.append(2)

print(castigatori.count(1))
print(castigatori.count(2))
p_j1 = castigatori.count(1)/1000
p_j2 = castigatori.count(2)/1000
print(f"primul are sansa de : {p_j1}")
print(f"Al doilea are sansa de : {p_j2}")


from matplotlib import pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD

model = BayesianNetwork([('Incepere', 'Jucator1'), ('Incepe', 'Jucator2')])


cpd_incepere = TabularCPD(variable='Incepere', variable_card=2, values=[[0.5], [0.5]])
cpd_jucator1 = TabularCPD(variable='Jucator1', variable_card=2, values=[[0.5, 0.5],
                                                                      [0.5, 0.5]],
                         evidence=['Incepe'], evidence_card=[2])

cpd_jucator2 = TabularCPD(variable='Jucator2', variable_card=2, values=[[0.5, 0.5, 0.5, 0.5],
                                                                    [0.5, 0.5, 0.5, 0.5]],
                        evidence=['Incepere', 'Jucator1'], evidence_card=[2, 2])
print(cpd_incepere)
print(cpd_jucator1)
print(cpd_jucator2)

model.add_cpds(cpd_incepere, cpd_jucator1, cpd_jucator2)
model.get_cpds()

model.check_model()


from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
posterior_p = infer.query(["incepere"], evidence={"Jucator1": 0.5})

#### 2
