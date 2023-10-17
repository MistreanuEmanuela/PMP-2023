from matplotlib import pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import network as nx


model = BayesianNetwork([('Cutremur', 'Incendiu'), ('Incendiu', 'Alarma'), ('Cutremur', 'Alarma')])


cpd_cutremur = TabularCPD(variable='Cutremur', variable_card=2, values=[[0.9995], [0.0005]])
cpd_incendiu = TabularCPD(variable='Incendiu', variable_card=2, values=[[0.99, 0.97],
                                                                      [0.01, 0.03]],
                         evidence=['Cutremur'], evidence_card=[2])
cpd_alarmă = TabularCPD(variable='Alarma', variable_card=2, values=[[0.9999, 0.05, 0.98, 0.02],
                                                                    [0.0001, 0.95, 0.02, 0.98]],
                        evidence=['Cutremur', 'Incendiu'], evidence_card=[2, 2])
print(cpd_cutremur)
print(cpd_incendiu)
print(cpd_alarmă)

model.add_cpds(cpd_cutremur, cpd_incendiu, cpd_alarmă)


assert model.check_model()

#pos = nx.circular_layout(model)
#nx.draw(model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
#plt.show()


from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
posterior_p = infer.query(["Cutremur"], evidence={"Alarma": 1})
print(posterior_p)

posterior_p = infer.query(["Incendiu"], evidence={"Alarma": 0})
print(posterior_p)



