# AIDD_Platform

This Platform is implemented by shiny express. The anaylsis code is python. 
This platform processes SMILES data to predict molecular properties and perform binary classification. You can download the analyzed results—including QED and SA scores—as a CSV file. 
If you want to perform prediction, set the Loss Function option in the Modeling sidebar to MSELoss; if you want to perform classification, select BCEWithLogitsLoss.

### Data Labeling
In your dataset, designate the SMILES column as "Smiles" and set the target variable to "Y".

### SMILES Preview
| Property             | Description                                                                                                                                                  |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **MW**               | Molecular Weight: the total mass of the molecule (g/mol), calculated by summing the atomic weights of all atoms.                                             |
| **LogP**             | Octanol–Water Partition Coefficient: the log₁₀ of the ratio of a compound’s concentration in octanol vs. water, indicating lipophilicity.                       |
| **HBA**              | Number of Hydrogen Bond Acceptors: count of atoms (usually oxygen and nitrogen) that can accept hydrogen bonds.                                               |
| **HBD**              | Number of Hydrogen Bond Donors: count of hydrogen atoms bound to electronegative atoms (O or N) that can donate hydrogen bonds.                               |
| **CSP3**             | Fraction of sp³ Carbons: proportion of carbon atoms that are sp³ hybridized, a measure of molecular saturation and 3D character.                               |
| **NumRotBond**       | Number of Rotatable Bonds: count of single, non-ring bonds around which free rotation can occur, reflecting conformational flexibility.                      |
| **NumRings**         | Ring Count: total number of distinct ring systems (aromatic and non-aromatic) in the molecule.                                                               |
| **TPSA**             | Topological Polar Surface Area: sum of surface areas of polar atoms (O, N) and their attached hydrogens, used to predict membrane permeability.              |
| **NumAromaticRings** | Number of Aromatic Rings: count of ring systems that follow Hückel’s rule and exhibit aromaticity.                                                          |
| **SAS**              | Synthetic Accessibility Score: heuristic score (1–10) estimating how easy (1) or difficult (10) a molecule is to synthesize.                                  |
| **QED**              | Quantitative Estimate of Drug-Likeness: composite metric (0–1) combining multiple properties (MW, LogP, H-bond counts, etc.) to estimate overall drug-likeness. |


### Data Preprocessing


### Modeling
| Option                  | Description                                                                                                                                           |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Hidden layers**       | The number of intermediate layers between the input and output layers where feature transformations occur.                                            |
| **Units per layer**     | A list specifying how many neurons each hidden layer contains (e.g. `64,64` means two hidden layers with 64 neurons each).                            |
| **Activation function** | The nonlinear function applied to each neuron’s output to introduce nonlinearity into the model.                                                      |
| **BatchNormalization**  | Whether to apply 1-dimensional batch normalization after each layer, which stabilizes and accelerates training by normalizing layer inputs.          |
| **Dropout rate**        | The fraction of neurons to randomly “drop” (ignore) during each training step to prevent overfitting (e.g., a rate of 0.5 drops 50% of units).       |
| **Loss function**       | The criterion used to measure the difference between the model’s predictions and the true target values.                                              |
| **MSELoss**             | Mean Squared Error Loss: computes the average of the squared differences between predicted and actual values, commonly used for regression.           |
| **BCEWithLogitsLoss**   | A loss function that combines a Sigmoid activation and binary cross-entropy in one step, operating on raw logits for improved numerical stability.    |
| **Optimizer**           | The algorithm that updates the network’s weights based on computed gradients to minimize the loss.                                                    |
| **Learning rate**       | The step size at which the optimizer updates the model parameters during training.                                                                   |
| **Epochs**              | The number of complete passes through the entire training dataset.                                                                                   |
| **Batch size**          | The number of samples processed before the model’s weights are updated once.                                                                         |


### Virtual Screening




