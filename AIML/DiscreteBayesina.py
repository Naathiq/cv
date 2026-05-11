from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.parameter_estimator import DiscreteMLE # Changed import
import numpy as np
import pandas as pd

# Define structure
model = DiscreteBayesianNetwork([('C', 'S'), ('D', 'S')])

# Generate sample data
data = pd.DataFrame(
    np.random.randint(0, 2, (5000, 3)),
    columns=['C', 'D', 'S']
)

# Fit the model using DiscreteMLE
model.fit(data, estimator=DiscreteMLE())

# Perform inference
infer = VariableElimination(model)

# Query probability of S given C=1
result = infer.query(['S'], evidence={'C': 1})

# Print result
print(result)
