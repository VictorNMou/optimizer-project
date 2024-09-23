import polars as pol
import pyomo.environ as pe

class Main:
    # Ler os dados
    data = pol.read_csv("./Data/knapsack.csv")
    data_m_w = pol.read_csv("./Data/max_weigth.csv")

    # Criar o modelo
    model = pe.ConcreteModel()

    # Conjunto de produtos
    model.products = pe.Set(initialize=data['Product'].unique().to_list(),
                            ordered=True,
                            doc="Set with all products")

    # Parâmetro de lucro para cada produto
    model.profit = pe.Param(model.products, 
                            domain=pe.NonNegativeReals, 
                            initialize=dict(zip(data['Product'], data['Value'])), 
                            doc="Profit Parameter for each product")

    # Parâmetro de peso para cada produto
    model.weight = pe.Param(model.products, 
                            domain=pe.NonNegativeReals,
                            initialize=dict(zip(data['Product'], data['Weight'])),
                            doc="Weight Parameter for each Product")

    # Parâmetro de peso máximo do carrinho
    model.max_weight = pe.Param(initialize=data_m_w['max_weigth'][0],
                                domain=pe.NonNegativeIntegers,
                                doc="Max weight Parameter of the cart")

    # Variáveis de decisão
    model.x = pe.Var(model.products,
                    domain=pe.NonNegativeIntegers,
                    doc="Decision variable of how much each product will enter the cart")

    # Restrição de peso
    def weight_constraint(model):
        return sum(model.x[p] * model.weight[p] for p in model.products) <= model.max_weight
    model.weight_constraint = pe.Constraint(rule=weight_constraint)

    # Função objetivo
    def objective(model):
        return sum(model.x[p] * model.profit[p] for p in model.products)
    model.obj = pe.Objective(rule=objective, sense=pe.maximize)

    # Exibir o modelo
    solver = pe.SolverFactory('glpk')
    results = solver.solve(model, tee = True)
    model.display()

if __name__ == "__main__":
    app = Main()