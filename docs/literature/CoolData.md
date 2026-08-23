# CoolData: Benchmarking Machine Learning Methods for Electronics Cooling

Anonymous Author(s)   
Affiliation   
Address   
email

## Abstract

The integration of Artificial Intelligence (AI) and Machine Learning (ML) into Computer-Aided Engineering (CAE) workflows promises significant speed-ups, particularly in the field of Computational Fluid Dynamics (CFD). However, the development of robust ML models in CFD is hindered by the lack of high-quality, application-specific datasets. This work addresses this gap with CoolData, a large-scale electronics cooling dataset containing 60, 848 stationary 3D flow and temperature fields for a diverse set of geometries, simulated using the industrial CFD solver Simcenter STAR-CCM+. To our knowledge, this is the first highquality and publicly available dataset for electronics cooling applications. We further show how this dataset can be leveraged in an ML pipeline with volumetric models (3D U-net) and surface centric models (MeshGraphNet, Transolver). Dataset is available at: https://huggingface.co/datasets/bgce/ cooldata-v2.

![](images/CoolData-28.jpg)  
Figure 1: Visualization of multiple data fields within a single sample from our dataset. The figure illustrates the temperature field (volume rendering, left), the velocity field (streamlines), and the turbulent kinetic energy (volume rendering, right).

## 1 Introduction

to address this challenge, including Neural Operator [3], Transformer-based [4, 5], and iterative solver-like architectures [6], and, most recently, Engineering Foundation Models (CAE-FMs)<sup>1</sup>.

The central bottleneck for Engineering AI, however, is data availability [7]. Unlike the text or image domains, engineering simulation data is scarce due to Intellectual Property (IP) constraints, must often be generated synthetically, and remains expensive to store and process [8, 7]. To put this in perspective: while the original OpenAI Codex model was trained on approximately 159 GB of Python code extracted from over 54 million GitHub repositories [9], a single high-fidelity CFD simulation can easily exceed 100 GB. Beyond sheer volume, existing public datasets are further limited in geometric and application diversity, predominantly covering simplified external aerodynamics geometries with only modest variation around selected base shapes [10, 11, 12, 13, 14]. While physics can often be well approximated by low-dimensional manifolds [15], geometry poses far greater challenges, as design spaces are in principle unbounded. This lack of geometric diversity restricts model generalizability and hinders the study of foundation model scaling behavior, a particularly pressing concern given recent evidence that foundation models regularly fail to generalize in physics when trained on insufficient or insufficiently diverse data [16].

Electronics cooling is a critical domain within CAE with broad relevance across electric and electronic systems, where thermal management frequently represents a key performance bottleneck [17]. Modern electronics are cooled under highly constrained geometrical conditions, with flow patterns dominated by wall-bounded turbulence, heat recirculation, and complex conduction paths [18, 19]. CFD and Conjugate Heat Transfer (CHT) simulations are the standard tool for this purpose, solving the Navier-Stokes and heat conduction equations simultaneously using turbulence models such as RANS [20]. Despite their physical fidelity, such simulations remain computationally expensive, making fast surrogate models highly desirable. Yet no large-scale, publicly available dataset exists to train and benchmark corresponding ML models in this domain.

We introduce CoolData, a large-scale, high-fidelity CFD dataset specifically designed for electronics cooling applications. CoolData comprises over 60, 000 stationary 3D thermal-fluid simulations for parametrically varied, geometrically complex configurations representative of real-world electronics cooling scenarios. The data is generated with the industrial solver Simcenter STAR-CCM+ [21]. To generate the data, we adopt a controlled geometry variation strategy based on up to 35 distinct parameters. This serves a dual purpose: it captures configurations typical of electronics cooling practice while enabling rigorous studies of data scaling behavior, making CoolData a valuable resource for investigating the data requirements of ML and CAE-FM. The dataset is published on Hugging Face (https://huggingface.co/datasets/bgce/cooldata-v2) under a CC BY-NC 4.0 license, with an accompanying accessor library (https://cooldata.readthedocs.io) supporting loading, visualization, and conversion for a wide range of ML pipelines.

Our main contributions are:

• A large-scale, open CFD dataset of 60, 848 high-fidelity 3D thermal-fluid simulations for geometrically diverse electronics cooling configurations (∼ 3 TB data in total).

• A controlled geometry variation strategy that enables systematic studies of data scaling behavior and the data requirements of ML and CAE-FM in an industrially relevant setting.

• ML benchmarks evaluating model accuracy as a function of data availability, using stateof-the-art volumetric architectures (3D U-Net [22]) and surface-centric architectures (Transolver [23], MeshGraphNet [24]), with a focus on in-distribution benchmarking.

• An open-source accessor library (https://cooldata.readthedocs.io) providing tools for loading, visualizing, and converting the dataset into formats suitable for a wide range of ML workflows.

## 2 Related Work

CFD Datasets for ML: Several open datasets have been released to support ML development for CFD, spanning a wide range of physical regimes and geometric complexity. Academic benchmarks 72 have traditionally relied on canonical geometries such as airfoils, bluff bodies, or duct flows, which are well understood and easy to validate [25, 26, 27]. Recently, multi-physics and multi-domain

![](images/CoolData-27.jpg)  
Figure 2: Dimensions and structure of the computational domain with 3 exemplary bodies.

74 benchmarks such as EAGLE [28] and The Well [8] have broadened the scope of available training data on academic examples significantly. Turbulence databases such as the Johns Hopkins Turbulence Database [29, 30] have provided foundational resources for the community. However, in many cases these are limited to very simple geometries. More recently, large-scale industrial-grade datasets have emerged, particularly in external aerodynamics: the Windsor body [31] and AhmedML [14] datasets offer high-fidelity CFD results for simplified road vehicle geometries; DrivAerML [12] and DrivAerNet++ [13] extend this to more realistic car shapes; and HiLiftAeroML [32] targets high-lift aircraft configurations using wall-modeled LES. Despite this progress, existing datasets focus almost exclusively on external aerodynamics or on academic flow configurations, with limited geometric variation and little coverage of internal thermal-fluid applications such as electronics cooling.

ML for Electronics Cooling: ML has shown increasing promise for thermal management applications, with studies demonstrating its utility for surrogate modeling of microchannel heat sinks, thermoelectric coolers, and phase-change material configurations [33, 34]. For example, Sikirica et al. [33] leveraged ML to predict thermal resistance and pumping power in microchannel heat sinks, significantly reducing computational cost compared to full CFD evaluations. However, industrial datasets in this space remain largely proprietary, and publicly available benchmarks with sufficient geometric diversity are scarce. Existing work predominantly relies on simplified geometries (such as rectangular channels, pin-fin arrays, or duct flows) which limits the complexity and diversity of systems that models are exposed to during training and, consequently, their generalizability to real-world scenarios where boundary conditions, materials, and geometries are far more varied.

Gap Addressed by CoolData: The scarcity of large, high-fidelity, and geometrically diverse datasets in the electronics cooling domain represents a significant barrier to developing generalizable ML models for thermal management. Most existing efforts are constrained either by limited geometry variation, small dataset scale, or restricted access due to IP concerns [7]. CoolData directly addresses this gap by providing a publicly available, large-scale dataset with systematic geometric variation, enabling not only the training of accurate surrogate models but also rigorous benchmarking of model performance as a function of data availability. Thus, this dataset provides a crucial step toward understanding the data requirements of foundation models in engineering [16].

## 3 Dataset

## 3.1 Baseline Geometry and Flow Conditions

This dataset focuses on (power-) electronics applications, comprising typical electronic components (such as chips, capacitors, coils, and transistors) mounted on a Printed Circuit Boards (PCBs) and enclosed within box-like housings. Thermal management is achieved through convective air cooling, a common and practically relevant cooling strategy. The specific configuration considered in the contribution is illustrated in Figure 2. Air enters the electronics compartment from the left (at x = 0) and exits on the right, modelled by constant, parametrized velocity at the inlet and a fixed pressure of zero at the outlet. The top and bottom boundaries are walls with a no-slip boundary condition with a wall model, whereas the right and left have symmetry boundary conditions, enabling to address a much larger, symmetrical domain than the size of the computational domain, which is more realistic for electronics cooling scenarios. Electronic components are geometrically abstracted as simple cuboids or cylinders mounted on the bottom wall. To promote a numerically stable and well-developed outflow, all components are positioned toward the inlet side of the domain, away from the outlet.

![](images/CoolData-35.jpg)  
Figure 3: Sample geometries from our dataset

## Parametrization and Generation of Geometry Variants

The domain for all samples in the dataset is a channel of fixed size as introduced above. Inside the domain, there are between 2 and 6 heated, wall-mounted bodies, either in a cuboid or cylindrical shape. While cuboids are often used to simulate PCB components, since they generate most of the flow features exhibited in PCB cooling, the addition of cylinders allows for the approximation of other components like capacitors and helps generalization. The bodies’ positions, sizes, and independent temperature are parametrized. This results in a total variation of maximal 35 parameters (depending on the number of involved geometries) for geometry, heating, and inflow (c.f., Table 1).

<table><tr><td>Parameter</td><td>Lower Bound</td><td>Upper Bound</td></tr><tr><td>Temperature  $\overline { { T _ { \mathrm { B o d y } } } }$ </td><td> $\overline { { 2 0 ~ } ^ { \circ } C }$ </td><td> $\overline { { 8 0 ~ } ^ { \circ } C }$ </td></tr><tr><td>Inlet velocity  $\vec { U } _ { \mathrm { i n } }$ </td><td> $\mathrm { 1 m s ^ { - 1 } }$ </td><td> $7 \mathrm { m } \mathrm { s } ^ { - 1 }$ </td></tr><tr><td>Position x</td><td>0.05 m</td><td>0.299 m</td></tr><tr><td>Position y</td><td>0m</td><td>0.099 m</td></tr><tr><td>Radius</td><td>0.001 m</td><td>0.1 m</td></tr><tr><td>Size  $s _ { x }$ </td><td>0.001 m</td><td>0.299 m</td></tr><tr><td>Size  $s _ { y }$ </td><td>0.001 m</td><td>0.099 m</td></tr><tr><td>Size  $s _ { z }$ </td><td>0.001 m</td><td>0.018m</td></tr></table>

Table 1: Bounds for each parameter. Size x and Size y apply to cuboids, Radius to cylinders only. All other geometry attributes are shared between the 2 body types.

Latin Hypercube Sampling [35] with equal probabilities is used to generate the parameter combinations. The number of bodies in the domain is variable, to cover simpler as well as more complicated cases. Our sample generation algorithm includes a minimum of 2 bodies, and each additional body is included with a probability of 80%. The 2 initial bodies are always cuboids, and the other potential bodies are 2 cuboids and 2 cylinders. After the designs are generated, we filter out those with bodies extending too close to the outflow boundary, as these designs usually take very long to converge. Thus effectively we consider only domains of half the size, since the outflow part is prolonged only for convergence reasons. Since more interesting geometries can be created when bodies overlap, we allow them to merge in this case. Examples of resulting geometries are presented in Figure 3.

## CFD Workflow

For all geometries we use the same CFD model. That is, we employ a steady state Reynolds-averaged Navier–Stokes equations (RANS) model with a $k - \varepsilon$ turbulence model and an appropriate wall layer model to appropriately reflect the fluid flow. The thermal behavior is modeled by means of a segregated fluid energy model.

The model is implemented in Simcenter STAR-CCM+ and the corresponding .sim-file is provided along the dataset. For further details, such as choice of specific solver parameters we refer to the provided .sim-file in the dataset and documentation in the user manual [21].

Meshing: For generating the mesh, we use a trimmed cell mesher based on hexahedral cells with a base size of 2 mm. Close to the boundaries, we use 3 prism layers with total thickness of 1 mm and stretching factor of 1.5. The total cell count varies between geometries, typically around $1 . 6 \times 1 0 ^ { 5 }$

To ensure that these mesh parameters allow for sufficient simulation accuracy while keeping computational cost and file size manageable, we conducted a mesh sensitivity test. The test compares the values of 4 typical quantities of interest (heat transfer at the outflow boundary, average velocity, average temperature, and maximum temperature) for different numbers of cells. If increasing the number of cells does not significantly change the quantities of interest, the mesh size is considered sufficient. In order to account for the variety of designs, the test was conducted on 3 different designs. Choosing finer grids, the relative variations in the quantities of interest across mesh configurations are small, with most coefficients of variation in the order of $1 . 0 \times 1 0 ^ { - 5 } \mathrm { t o } 1 . 0 \times 1 0 ^ { - 7 }$ . Therefore, we considered the chosen mesh settings to be optimal for the purpose of the dataset.

Convergence Criteria: Our dataset focuses on steady flows. When the main quantities of interest, i.e., average velocity, average temperature, and heat transfer at the outflow boundary, exhibit asymptotic behavior the corresponding simulation is considered to be converged. We use | max − min | over the last 30 iterations as a criteria, i.e., stop the simulation when these have reached $1 . 0 \times 1 0 ^ { - 4 }$ $1 . 0 \times 1 0 ^ { - 5 } .$ , and $1 . 0 \times 1 0 ^ { - 4 }$ , respectively. Residuals are expected to decrease during convergence but may vary in magnitude across different designs. To address this, a loose minimum criterion of 0.01 is imposed on the normalized residuals to prevent divergence, while a standard deviation criterion of 0.01 over the last 100 iterations ensures residuals stabilize before stopping the simulation. These criteria filter out designs with large, oscillatory residuals characteristic of transient flows.

Given the Reynolds-numbers of flow regimes, corresponding flows will not be quasi-stationary leading to high computational efforts resolving corresponding time series. Thus in industrial practice typically a RANS model is solved instead of relying on Direct Numerical Simulation (DNS) of the Navier–Stokes equations. But even using a RANS model, mathematically, a steady flow cannot be guaranteed for all cases. Therefore, designs with significant transient behavior need to be excluded, as these fail to reach a steady state solution within a reasonable number of iterations. To balance the risk of discarding valid designs that require more iterations with the computational cost of simulating transient designs, we disregard simulations which have not converged after 1500 iterations.

Computational Resources: Simulations were carried out on the CoolMUC-4 cluster<sup>2</sup>, which is operated by the Leibniz Supercomputing Centre, an institute of the Bavarian Academy of Sciences and Humanities, using the commercial CFD solver Simcenter STAR-CCM+<sup>3</sup>. For each data point, a Simcenter STAR-CCM+ .sim file was generated, which contained a parameterized representation of the geometry. A macro was then executed within Simcenter STAR-CCM+ to automate the entire workflow. This macro was responsible for setting the parameters to the values specified in a configuration file, generating the computational mesh, running the simulation to convergence, and exporting the final results into the .cgns format. CFD General Notation System (CGNS)<sup>4</sup> is the most common general, portable, and extensible standard for storing and retrieving CFD analysis data, specifically it can be comfortably accessed via Paraview [36] for post processing.

## Dataset Contents and Availability

All data points in the dataset have a unique ID, ranging from 1 to 1, 000, 000 used to identify the corresponding file names. As we only include converged cases, some IDs are absent from the .cgns files. The dataset consists of 60,848 data points, with a total compressed size of approximately $\sim 1 \mathrm { T B }$ (995GB as distributed on Hugging Face), corresponding to an uncompressed size of approximately ∼ 3 TB. For each data point we provide the following data:

• Geometry parameters for each cases provided in the metadata.parquet $\mathrm { \ f l e ^ { 5 } }$ . Based on these parameters all geometries are fully described.

• Volume Field Quantities for each case covering the 3D vector velocity field ${ \vec { v } } ( { \vec { x } } ) [ \mathrm { m } / \mathrm { s } ]$ the static pressure field $P ( \vec { x } ) \left[ \mathrm { P a } \right]$ , the fluid temperature distribution $T ( \vec { x } ) \ [ \mathrm { K } ]$ as well as the specific turbulent kinetic energy $k ( \vec { x } ) [ \mathrm { m } ^ { 2 } / \mathrm { s } ^ { 2 } ]$ and the dissipation rate of turbulent kinetic energy $\varepsilon ( \vec { x } ) [ \mathrm { m } ^ { 2 } / \mathrm { s } ^ { 3 } ]$

• Wall-related surface quantities for each surface (per surface data), covering surface area $[ \mathrm { m } ^ { 2 } ]$ , normal, wall temperature (K), heat transfer coefficient $[ \mathrm { W } / ( \mathrm { m } ^ { 2 } \cdot \mathrm { K } ) ]$ ], static pressure $\mathrm { [ P a ] }$ , wall shear stress vector [Pa] associated with each wall element.

The data is organized in set of runs, where each run is a compressed .zip archive. Each archive stores a batch of 10 samples as a flat list of .cgns files, with a separate file for surface and volume data, i.e., each batch consists of 20 files in total. (A more detailed description of the stored quantities can be found in Appendix G).

The dataset is publicly available on Hugging Face with a permissive CC BY-NC 4.0 license at https://huggingface.co/datasets/bgce/cooldata-v2 (see https://doi.org/ 10.57967/hf/5744) and can be accessed via a dedicated Python library with intuitive interface to load the data, retrieve metadata, and visualize samples (see https://cooldata.readthedocs.io and https://pypi.org/project/cooldata/). We provide detailed metadata for the dataset using the Croissant format [37].

The dataset provides a comprehensive set of typical quantities commonly used in CFD applications such as flow prediction or heat transfer modeling (c.f., Appendix G). Specifically they allow the calculation of further derived quantities (e.g., pressure coefficient, total pressure coefficient, skin friction coefficient, temperature coefficient, or heat flux) or integral quantities (e.g., total pressure drop, component junction temperatures, thermal resistance, or fan operating point parameters). Figure 4 shows a comparative analysis of selected hydraulic performance metrics such as the heat dissipated, mean HTC coefficient (across all surfaces), inlet velocity, and maximum temperature.

## 4 Benchmarks

Given the size of the dataset, it provides a unique opportunity to benchmark different models not only in terms of their performance but also in terms of their data requirements. Within this contribution we use the dataset to benchmark 3 representative ML models which have been broadly used in the context of Engineering: a volumetric model and 2 surface-centric models. We thereby focus on field prediction rather than prediction of derived or integrated quantities. The latter can be determined using appropriate post-processing from the predicted field quantities (c.f., Appendix G). Since the dataset addresses thermal management, we consider the temperature field as the primary quantity of interest.

## ML-based field prediction from 3D meshes using geometric deep learning

As a first benchmark we evaluate the capability of 3 different deep learning models to predict physics field quantities. We consider 3D U-Net [38] as a typical volumetric model as well as MeshGraph-Net [39] and Transolver [23] as typical representatives of surface-centric models. 3D U-Net is implemented via pytorch-3dunet [40] whereas Transolver and MeshGraphNet are implemented via NVIDIA PhysicsNeMo [24]. For more details on the specific training we refer to Appendix E. The obtained performance metrics are shown in Table 2 and selected results are shown in Figure 5. For both, results have been obtained using 50, 000 samples over 100 epochs on 8×NVIDIA H100 GPUs. Inference times are reported per sample as the average over 100 runs on a CPU (AMD64 Ryzen, AuthenticAMD), with all 3 models evaluated on identical hardware to ensure comparability.

<table><tr><td>Model</td><td>MSE</td><td>MAE</td><td>Max AE</td><td> $\overline { { \mathbf { R } ^ { 2 } } }$ </td><td> $\overline { { \mathbf { T } _ { \mathrm { T r a i n i n g } } } }$ </td><td> $\mathbf { T } _ { \mathrm { I n f e r e n c e } }$ </td><td># Parameters</td></tr><tr><td>3D U-Net [38]</td><td>0.00549</td><td>2.07℃</td><td>58.46℃</td><td>0.8156</td><td>38 mins</td><td>440.3 ms</td><td>3,931,047</td></tr><tr><td>MeshGraphNet [39]</td><td>0.02109</td><td>5.73℃</td><td>59.56℃</td><td>0.5034</td><td>40 mins</td><td>8362.6 ms</td><td>2,339,201</td></tr><tr><td>Transolver [23]</td><td>0.00122</td><td>0.82℃</td><td>60.24℃</td><td>0.9685</td><td>1 hour</td><td>1369.1 ms</td><td>1,539,649</td></tr></table>

Table 2: Comparative analysis of Deep Learning models for thermal field prediction with 50, 000 samples. The corresponding error is evaluated on 1, 000 in-distribution test samples.

232 The results in Table 2 highlight a clear hierarchy in predictive performance across the 3 architectures.   
233 Transolver achieves the best results across all metrics, with $\mathrm { M S E } { = } 0 . 0 0 1 2 2$ $\mathrm { M A E { = } 0 . 8 2 ^ { \circ } C }$ , and   
234 $R ^ { 2 } { = } 0 . 9 6 8 5$ . A qualitative comparison of the predicted surface temperature field against the CFD   
235 ground truth is shown in Figure 5, where the error is largely confined to component edges and corners.   
3D U-Net achieves236 $R ^ { 2 } { = } 0 . { 8 1 5 6 }$ and $\mathrm { M A E } { = } 2 . 0 7 ^ { \circ } \mathrm { C }$ at substantially lower training time (38 mins)

![](images/CoolData-01.jpg)  
Figure 4: Pair plot of key thermal performance metrics in the dataset. The diagonal histograms show the marginal distributions of total heat dissipated $Q _ { \mathrm { t o t a l } } .$ , mean component heat transfer coefficient $\bar { h } _ { \mathrm { c o m p } }$ , inlet velocity $U _ { \mathrm { i n l e t } }$ , and maximum component temperature $T _ { \mathrm { m a x } }$ . The off-diagonal scatter plots visualize pairwise relationships between these quantities, revealing the trade-offs between cooling performance, thermal safety, and flow conditions. Colors denote geometry categories based on body count, illustrating how geometric complexity influences the occupancy of the performance space and highlighting the diversity of configurations represented in the dataset.

and the fastest inference (440 ms). MeshGraphNet underperforms both alternatives $( R ^ { 2 } { = } 0 . 5 0 3 4$ $\mathrm { M A E } { = } 5 . 7 3 ^ { \circ } \mathrm { C } )$ , a consequence of its limited message-passing receptive field relative to the graph diameter of the domain, as discussed further in the scaling analysis below.

## Data-Accuracy relationship for geometric deep learning models

Having a massive dataset of over 60, 000 simulations, the dataset offers a unique opportunity to study the scaling behavior of the different models with number of data points. Using the same setup as above but varying the number of data points, the test and training error are shown in Figure 6.

The results reveal 3 qualitatively distinct scaling regimes, consistent with the architectural properties of each model. Transolver exhibits the strongest data dependence, with test MSE decreasing by nearly 2 orders of magnitude as the training set grows from 1, 000 to 50, 000 samples (0.034 to 0.001). The narrowing gap between training and test loss at higher sample counts indicates improved generalisation, suggesting that the model has not yet saturated and could further benefit from more data.

The 3D U-Net demonstrates strong data efficiency, achieving a test MSE below 0.010 even when trained with only 1, 000 samples. However, the model’s performance exhibits diminishing returns beyond approximately 10, 000 samples, where the training loss continues to decrease while the test loss plateaus. This train–test divergence is characteristic of models constrained by a resolution ceiling rather than by capacity or data limitations. Once voxelisation error becomes the dominant source of error, further increases in data size no longer translate into improved generalisation.

![](images/CoolData-02.jpg)  
Figure 5: Comparison of the input 3D geometry, corresponding CFD ground truth, and ML predictions for the Transolver model. The visualization shows surface temperature fields predicted by the surrogate model against CFD reference data, along with pointwise absolute error highlighting regions of largest discrepancy at component edges and corners.

![](images/CoolData-17.jpg)  
Figure 6: Scaling of the loss (MSE) on log scale of the different deep learning models depending on the number of available data points.

MeshGraphNet exhibits near-flat scaling behaviour across all dataset sizes, with both training and test losses remaining within a narrow range ( 0.015–0.022) regardless of sample count. We investigated whether increasing the number of message-passing steps or hidden dimensionality could alleviate this limitation, but found no significant improvement, suggesting a fundamental architectural bottleneck: fixed-radius message passing limits the ability to capture long-range thermal dependencies in the elongated channel geometry. As a result, increasing the dataset size does not improve performance. This behaviour is consistent with known limitations of message-passing graph neural networks, where performance can suffer when depth is increased [41].

## 264 5 Conclusion

In this contribution, we present CoolData: a large-scale electronics cooling dataset containing over 60,000 stationary 3D flow fields for a diverse set of geometries. Additionally, we share two example ML workflows, one using voxelized volumetric data and the other using point-based / graph-based surface representations, to demonstrate how to use the dataset for training ML models.

Our dataset is a valuable contribution to the race towards CAE-FM through its volume and diversity. However, we expect that a true CAE-FM will require further expansion, incorporating additional geometries, boundary conditions, flow regimes, and solvers.

• New Dataset Contribution: We present a novel large-scale dataset for electronics cooling applications, generated using the industrial-grade CFD solver Simcenter STAR-CCM+. To the best of our knowledge, this is the first publicly available dataset of this scale and fidelity targeting the domain of electronics cooling and thermal management.

• Thermal Management as an Emerging Domain: While existing industrial-grade datasets focus predominantly on external aerodynamics, CoolData dataset targets electronics cooling - a domain of rapidly growing relevance. With the electrification of many industries, thermal management is increasingly becoming a critical limiting factor across sectors. Although external aerodynamics remains one of the most prominent application domains for CFD today, we anticipate that thermal management use cases will become comparably, if not more, important in the coming years.

• Geometric Richness: A key feature of CoolData is its geometric richness. In contrast to existing datasets, which are often constrained to small parametric variations of a base geometry, CoolData features substantial structural diversity. This variability arises from the distribution of electronic components on PCBs, making it a more challenging benchmark for machine learning models.

## 288 6 Limitations and Future Work

We see two major limitations concerning the current dataset: many real-world applications are not necessarily "close" to the data distribution presented and more realistic simulation methods are available, yet infeasible for a dataset of this scale.

While the scenario and data distribution adopted in this contribution (c.f., Section 3) is very rich, many real electronics components and cooling scenarios will still fall outside this distribution. Thus many more geometries would be required to realize a proper CAE-FM.

Following industry practice we focus on a stationary RANS model. However, this is an approximation of more accurate solution strategies allowing computational feasibility. Transient solutions found by methods like Large Eddy Simulation (LES) and Detached Eddy Simulation (DES) would be more realistic. These methods would also alleviate geometry constraints like the long free region after the random geometry needed to ensure a recirculation free exit plane. Yet the amount of data produced when storing individual states of the transient simulation for all geometries would be hundreds of times larger and thereby become infeasible to store and distribute. When condensing the states into average quantities to be stored, those simulation methods are orders of magnitude more expensive to compute, so using the same amount of compute resources, only a much lower number of scenarios could have been simulated.

Last but not least, we limit our benchmarking to three standard ML models frequently used in the context of Computer Aided Engineering (CAE) as well as testing the method only on in-distribution cases. In the future, we plan to test more ML models as well as explore out-of distribution generalization, e.g., by considering electronic shapes differently than cylinders and cuboids such as pyramids or trapezoidal shapes. Based on these investigations we intend to further substantiate estimations of data requirements towards industrial CAE-FM.

Acknowledgements: The authors thank Robin Bornoff and Alexandru Ciobanas from Siemens Digital Industries Software for their kind introduction to and support in using Simcenter STAR-CCM+. The authors furthermore acknowledge the support of the Leibniz Supercomputing Centre, operated by the Bavarian Academy of Sciences and Humanities, for providing HPC resources, as well as Hugging Face, Inc., for provision of the cloud storage. Most of the results have been generated in the context of the project course Towards Foundational CAE Models of the Bavarian Graduate School of Computational Engineering (BGCE).

Contributions: DH: conceptualization and methodology; FD, EC, JH, OP, BP, DS: software development and data generation; SG/FD: surrogate modeling; FD/SG/DH: writing - review and editing; All authors read and approved the final manuscript.

## References

[1] Jan Paul Stein and Alessandro Faure Ragani. On the brink of a revolution? Engineering simulation in the age of AI, 2025.

[2] Engineering.com. Research Report: The State of Simulation, Prototyping and Validation. Technical report, Engineering.com, April 2022.

[3] Maximilian Herde, Bogdan Raonic, Tobias Rohner, Roger Käppeli, Roberto Molinaro, Em-´ manuel de Bézenac, and Siddhartha Mishra. Poseidon: Efficient Foundation Models for PDEs, May 2024.

[4] Benedikt Alkin, Andreas Fürst, Simon Schmid, Lukas Gruber, Markus Holzleitner, and Johannes Brandstetter. Universal physics transformers: A framework for efficiently scaling neural operators. Advances in Neural Information Processing Systems, 37:25152–25194, 2024.

[5] Huakun Luo, Haixu Wu, Hang Zhou, Lanxiang Xing, Yichen Di, Jianmin Wang, and Mingsheng Long. Transolver++: An accurate neural solver for PDEs on million-scale geometries. arXiv preprint arXiv:2502.02414, 2025.

[6] Rishikesh Ranade, Mohammad Amin Nabian, Kaustubh Tangsali, Alexey Kamenev, Oliver Hennigh, Ram Cherukuri, and Sanjay Choudhry. Domino: A decomposable multi-scale iterative neural operator for modeling large scale engineering simulations. arXiv preprint arXiv:2501.13350, 2025.

[7] Neil Ashton, Johannes Brandstetter, and Siddhartha Mishra. Fluid Intelligence: A Forward Look on AI Foundation Models in Computational Fluid Dynamics, November 2025.

[8] Ruben Ohana, Michael McCabe, Lucas Meyer, Rudy Morel, Fruzsina Agocs, Miguel Beneitez, Marsha Berger, Blakesly Burkhart, Stuart Dalziel, Drummond Fielding, et al. The Well: a large-scale collection of diverse physics simulations for machine learning. Advances in Neural Information Processing Systems, 37:44989–45037, 2024.

[9] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

[10] Xiaoxiao Guo, Wei Li, and Francesco Iorio. Convolutional neural networks for steady flow approximation. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pages 481–490, 2016.

[11] Keefe Huang, Moritz Krügener, Alistair Brown, Friedrich Menhorn, Hans-Joachim Bungartz, and Dirk Hartmann. Machine learning-based optimal mesh generation in computational fluid dynamics. arXiv preprint arXiv:2102.12923, 2021.

[12] Neil Ashton, Charles Mockett, Marian Fuchs, Louis Fliessbach, Hendrik Hetmann, Thilo Knacke, Norbert Schonwald, Vangelis Skaperdas, Grigoris Fotiadis, Astrid Walle, Burkhard Hupertz, and Danielle Maddix. DrivAerML: High-fidelity computational fluid dynamics dataset for road-car external aerodynamics, 2025.

[13] Mohamed Elrefaie, Florin Morar, Angela Dai, and Faez Ahmed. DrivAerNet++: A largescale multimodal car dataset with computational fluid dynamics simulations and deep learning benchmarks. Advances in Neural Information Processing Systems, 37:499–536, 2024.

[14] Neil Ashton, Danielle Maddix, Samuel Gundry, and Parisa Shabestari. AhmedML: High-fidelity computational fluid dynamics dataset for incompressible, low-speed bluff body aerodynamics. arxiv.org, 2024.

[15] Alfio Quarteroni, Paola Gervasio, and Francesco Regazzoni. Combining physics-based and data-driven models: advancing the frontiers of research with scientific machine learning. arXiv preprint arXiv:2501.18708, 2025.

[16] Keyon Vafa, Peter G. Chang, Ashesh Rambachan, and Sendhil Mullainathan. What has a foundation model found? Using inductive bias to probe for world models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 60727–60747. PMLR, 7 2025.

[17] Packy McCormick and Sam D’Amico. The electric slide: The energy transition is happening faster than you think. https://www.notboring.co/p/the-electric-slide, May 2025. Not Boring Newsletter.

[18] Mustafa Emad, Ahmed Abdulnabi, and Sattar Aljabair. Heat transfer in electronic system printed circuit board: A review. Engineering and Technology Journal, 40:99–108, 01 2022.

[19] E.R. Meinders and K. Hanjalic. Experimental study of the convective heat transfer from in-line´ and staggered configurations of two wall-mounted cubes. International Journal of Heat and Mass Transfer, 45(3):465–482, 2002.

[20] Vishwas Kulkarni and Alexander Kospach. Machine learning based reduced order model for full temperature prediction of power electronics component cooling. In 2024 Energy Conversion Congress & Expo Europe (ECCE Europe), pages 1–8. IEEE, 09 2024.

[21] Siemens Digital Industries Software. Simcenter STAR-CCM+, February 2025. Version 2402.0001 (19.02.013). Multiphysics CFD simulation software.

[22] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation, 2015.

[23] Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A fast transformer solver for PDEs on general geometries. arXiv preprint arXiv:2402.02366, 2024.

[24] PhysicsNeMo Contributors. Nvidia physicsnemo: An open-source framework for physics-based deep learning in science and engineering. https://github.com/NVIDIA/physicsnemo, February 2023.

[25] Kazuto Hasegawa, Kai Fukami, Takaaki Murata, and Koji Fukagata. Machine-learning-based reduced-order modeling for unsteady flows around bluff bodies of various shapes. Theoretical and Computational Fluid Dynamics, 34:367–383, 2020.

[26] Nils Thuerey, Konstantin Weißenow, Lukas Prantl, and Xiangyu Hu. Deep learning methods for reynolds-averaged navier–stokes simulations of airfoil flows. AIAA journal, 58(1):25–36, 2020.

[27] Florent Bonnet, Jocelyn Mazari, Paola Cinnella, and Patrick Gallinari. Airfrans: High fidelity computational fluid dynamics dataset for approximating reynolds-averaged navier–stokes solutions. Advances in Neural Information Processing Systems, 35:23463–23478, 2022.

[28] Steeven Janny, Aurélien Beneteau, Madiha Nadri, Julie Digne, Nicolas Thome, and Christian Wolf. Eagle: Large-scale learning of turbulent fluid dynamics with mesh transformers. arXiv preprint arXiv:2302.10803, 2023.

[29] Y. Li, E. Perlman, M. Wan, Y. Yang, C. Meneveau, R. Burns, S. Chen, A. Szalay, and G. Eyink. A public turbulence database cluster and applications to study lagrangian evolution of velocity increments in turbulence. Journal of Turbulence, 9(31), 2008.

[30] E. Perlman, R. Burns, Y. Li, and C. Meneveau. Data exploration of turbulence simulations using a database cluster. In Proceedings of the 2007 ACM/IEEE Conference on Supercomputing (SC07). ACM/IEEE, 2007.

[31] Neil Ashton, Jordan Angel, Aditya Ghate, Gaetan Kenway, Man Long Wong, Cetin Kiris, Astrid Walle, Danielle Maddix, and Gary Page. Windsorml: High-fidelity computational fluid dynamics dataset for automotive aerodynamics. Advances in Neural Information Processing Systems 37, 2024.

[32] Neil Ashton, Adam M. Clark, Christopher Ivey, Liam Heidt, Sanjeeb Bose, Rishi Ranade, Rahul Agrawal, and Konrad Goc. High-Fidelity CFD Data Generation for HiLiftAeroML using Solution-Adapted WMLES, page 0042. AIAA, 2026.

[33] Ante Sikirica, Luka Grbciˇ c, and Lado Kranj´ ceviˇ c. Machine learning based surrogate models for´ microchannel heat sink optimization, 2022.

[34] Vishal Anand and Mohan Sangeeth. Machine learning optimization to boost the effectiveness of phase change material (pcm)-based on-chip passive thermal management. Electronics Cooling Magazine, October 2022.

[35] Michael D McKay, Richard J Beckman, and William J Conover. A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics, 42(1):55–61, 2000.

[36] James Ahrens, Berk Geveci, and Charles Law. Paraview: An end-user tool for large data visualization. In Charles D. Hansen and Christopher R. Johnson, editors, The Visualization Handbook, pages 717–731. Academic Press / Elsevier, 2005.

[37] Mubashara Akhtar, Omar Benjelloun, Costanza Conforti, Luca Foschini, Pieter Gijsbers, Joan Giner-Miguelez, Sujata Goswami, Nitisha Jain, Michalis Karamousadakis, Satyapriya Krishna, et al. Croissant: A metadata format for ML-ready datasets. Advances in Neural Information Processing Systems, 37:82133–82148, 2024.

[38] Özgün Çiçek, Ahmed Abdulkadir, Soeren S Lienkamp, Thomas Brox, and Olaf Ronneberger. 3D U-Net: learning dense volumetric segmentation from sparse annotation. In International conference on medical image computing and computer-assisted intervention, pages 424–432. Springer, 2016.

[39] Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W Battaglia. Learning mesh-based simulation with graph networks. arXiv preprint arXiv:2010.03409, 2020.

[40] Adrian Wolny et al. pytorch-3dunet: 3D U-Net for volumetric segmentation, 2022.

[41] Zhanghao Wu, Paras Jain, Matthew Wright, Azalia Mirhoseini, Joseph E. Gonzalez, and Ion Stoica. Representing long-range context for graph neural networks with global attention. In Advances in Neural Information Processing Systems, volume 34, pages 13266–13279. Curran Associates, Inc., 2021.

[42] Adrian Wolny, Lorenzo Cerrone, Athul Vijayan, Rachele Tofanelli, Amaya Vilches Barro, Marion Louveaux, Christian Wenzl, Sören Strauss, David Wilson-Sánchez, Rena Lymbouridou, et al. Accurate and versatile 3d segmentation of plant tissues at cellular resolution. Elife, 9:e57613, 2020.

## A Broader Impact

Contrary to many aerodynamic datasets, CoolData dataset provides a much richer diversity of geometries, making it more readily generalizable to other domains. At the same time the diversity makes it much harder to train compared to other datasets. Therefore, it constitutes a valuable resource for multiple domains, applications, and research communities in the following ways:

Despite its relatively compact geometric primitives, the dataset’s structure allows it to be directly applied to selected real-world use cases, such as thermal management of PCB boards in rack-blade servers and similar computer enclosures.

• Improving Electronics Cooling Applications: The core focus of this dataset is the improvement of electronics cooling applications, specifically increasing their cooling efficiency. Despite its relatively compact geometric primitives, the dataset’s structure allows it to be directly applied to selected real-world use cases, such as blade servers in datacenters. Since a significant portion of a datacenter energy consumption is used for cooling, CoolData will have direct impact on the energy needs of datacenters by enabling faster design iterations aiming for higher cooling efficiencies.

• Accelerating CFD Simulations: By providing high-fidelity simulation data, CoolData furthermore enables the training of surrogate models that substantially reduce the computational cost and time required for CFD applications well beyond electronics cooling. For example, wind simulation in cities follows similar geometries and thus the data will also impact ML models in this space. This facilitates faster design iterations in many more domains. Thereby, surrogate models may either fully replace traditional simulations or serve as an informed initial guess within an iterative solver.

• Benchmarking and Validation: The dataset introduces thermal management as a new benchmark domain for training and testing advanced ML models, including geometric Deep Learning (DL) approaches. It thus supports the validation and systematic comparison of different ML modeling strategies beyond external aerodynamics, which is the primary focus of benchmarking CAE ML models today.

• Machine Learning Integration: CoolData dataset offers a rich source for training and evaluating advanced machine learning models on complex conjugate heat transfer flows, supporting the broader integration of data-driven methods into engineering workflows.

• Cross-Disciplinary Insights: Combined with existing CFD datasets, CoolData dataset enables cross-disciplinary insights at the intersection of different flow regimes and engineering applications, fostering collaboration across research communities.

Therefore, the large dataset of 60848 electronics cooling designs is expected to drive significant innovations and advancements in both academic and industrial settings. In particular, given that the dataset reflects industrial realism through the use of an industrial-grade solver, the methodologies developed here are not necessarily limited to thermal management of electronic systems. They can be extended to thermal management applications more broadly, and even beyond, into external aerodynamics and other domains involving numerical simulations (specifically CFD) and DL.

(a)  
![](images/CoolData-03.jpg)

(b)  
![](images/CoolData-32.jpg)

(c)  
![](images/CoolData-26.jpg)  
Figure 7: Distribution of heights of geometries across all samples: (a) average height, (b) variance, and (c) frequency.

(a)  
![](images/CoolData-08.jpg)

(b)  
![](images/CoolData-33.jpg)

![](images/CoolData-13.jpg)

![](images/CoolData-10.jpg)

![](images/CoolData-15.jpg)  
Figure 8: Principal component analysis across the 50, 000 samples of the dataset.

## 487 B Geometry Variations

The dataset consists of two to six random bodies distributed across the domain. Thereby, each body is either a cylinder or a rectangular box with position, width and height varied randomly (c.f., Section 3). To investigate the variability of the data we visualize their average distribution, the corresponding variability as well as the corresponding correlations.

For each geometry we calculate the corresponding height field, that is the field $h ( x , y ) \to \mathbb { R }$ which maps to each position the height of the corresponding body. Here a height of 0 corresponds to no body being present at the specific position. Since the simulation considers a boolean combination of bodies, the height field will always indicate the maximum height, if multiple bodies are present.

In Figure 7, we visualize the mean height field $\begin{array} { r } { \bar { h } ( x , y ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { h _ { i } ( x , y ) } } \end{array}$ , the variance $h _ { \sigma ^ { 2 } } ( x , y ) =$ $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \bar { h } ( x , y ) - h _ { i } ( x , y ) ) ^ { 2 } } \end{array}$ , and the occupancy frequency $\begin{array} { r } { h _ { f r e q } ( x , y ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } [ h _ { i } ( x , y ) > 0 ] } \end{array}$ using the Iverson bracket notation, i.e., $[ x > 0 ] \stackrel { \cdot } { = } 1 \mathrm { i f } \stackrel { - } { x } > \stackrel { \cdot } { 0 }$ and 0 otherwise. As seen in Figure 7, we can observe the distribution as expected from the random sampling adopted here (c.f., Section 3). That is, the mean field is higher in the middle since here geometries with center points above or below the middle contribute while closer to the boundaries there is only one sided contributions, since there are no geometries with center points outside the flow channel.

To further investigate variability of the dataset, a Principal Component Analysis has been performed on set of the height functions $h _ { i } ( x , y ) , 1 \leq i \leq N$ after an appropriate discretization. The distribution of corresponding eigenvalues and the first four modes are shown in Figure 8. The low slow decay of the eigenvalues as well as the major eigenmodes indicate a homogeneous distribution of geometries as expected.

## C Sample Gallery

Figure 9 shows a set of 16 sample geometries from the dataset along with the visualization of different physical field variables of interest, i.e., velocity fields (streamlines), pressure fields, temperature field, and turbulent kinetic energy.

![](images/CoolData-18.jpg)  
Figure 10: Per-sample relationship between mean and maximum wall HTC, coloured by the maximum body set temperature

![](images/CoolData-22.jpg)  
Figure 9: 16 different geometry samples from the dataset with visualization of velocity fields (streamlines), pressure fields, temperature field, and turbulent kinetic energy (columns from left to right)

(a)  
![](images/CoolData-34.jpg)

(b)  
![](images/CoolData-21.jpg)  
Figure 11: Spatial Distribution of Heat Transfer Coefficients and their variation.

![](images/CoolData-30.jpg)

![](images/CoolData-12.jpg)

![](images/CoolData-06.jpg)

![](images/CoolData-23.jpg)

![](images/CoolData-29.jpg)

![](images/CoolData-16.jpg)  
Figure 12: Distribution of key surface quantities (Total Heat Dissipated, Mean and Max HTC, as well as Mean, Max, Min body temperature) across samples.

## D In-Depth Dataset Insights

In Figure 10, the scatter plot reveals that the majority of samples cluster at low mean HTC values (below 500 W/m²K), consistent with realistic forced convection conditions in electronics cooling applications, while maximum HTC spans a much wider range up to 10, 000 W/m²K, indicating highly localized heat transfer hotspots. The colour encoding shows a clear positive correlation between body temperature and both mean and maximum HTC, confirming that higher thermal loads drive more intense convective transfer. This distribution demonstrates the dataset’s physical consistency and the challenge it poses for surrogate models: capturing rare but physically significant high-HTC outliers alongside the dense low-HTC majority.

In Figure 11 we consider the spatial distribution of the heat transfer coefficient. This nicely correlates with the geometry distribution as shown in Figure 7 as expected.

Figure 12 shows the distribution of key surface quantities across the dataset. The Max HTC and Mean HTC distributions are right-skewed, with most samples concentrated at lower values and a long tail of high-intensity outliers, consistent with the scatter plot in Figure 10. The Total Heat Dissipated distribution similarly peaks at low values. The body temperature distributions (mean, max, min) span approximately 290-355 K with a relatively uniform spread, indicating good coverage of the operating temperature range and avoiding bias toward any particular thermal regime.

## E Deep Surrogate Models

The provided dataset is benchmarked on three different DL models, namely 3D U-Net [42], MeshGraphNet [39], and Transolver [23]. To do so, we rely on openly available implementations of the corresponding models, namely NVIDIA’s PhysicsNeMo [24], and the 3D U-Net via pytorch-3dunet [40].

## 3D U-Net

We evaluate a 3D U-Net [42] as a grid-based baseline for temperature field prediction. Unlike point-based neural operators, the model operates on a structured volumetric representation of the physical domain.

Problem formulation: The input domain is discretized into a regular grid of resolution $6 4 ^ { 3 }$ and the model learns a mapping

$$
\mathbf { X } \in \mathbb { R } ^ { 3 \times D \times H \times W } \to \mathbf { Y } \in \mathbb { R } ^ { 1 \times D \times H \times W } ,
$$

where $D = H = W = 6 4$ . The input consists of multiple physically meaningful channels defined on the voxel grid (see below). Being based on a volumetric discretization, 3D U-Net is effectively a volume model.

Input representation: The input tensor is constructed from three channels:

• Signed Distance Field (SDF): encodes geometry of solid bodies (cuboids and cylinders), providing both inside/outside information and distance to boundaries.

• Boundary temperature field: voxelized heat source temperatures mapped onto the grid.

• Inflow velocity field: a global boundary condition injected as a spatially constant slice.

Together, these form a 3-channel volumetric representation:

$$
\mathbf { X } = [ \mathrm { S D F } , \ T _ { \mathrm { b o u n d a r y } } , \ u _ { \mathrm { i n f l o w } } ] .
$$

Geometry and physics encoding: Geometric information is encoded via a signed distance field computed analytically from cuboids and cylinders. Boundary conditions are projected onto the grid using occupancy-based voxelization.

Model architecture: We use a 3D U-Net with 3 encoding levels and a base feature width of 32, doubling at each successive level (32, 64, 128). The network takes a 3-channel volumetric input and produces a single-channel temperature field output. Standard convolutional downsampling with skip connections is used to preserve spatial detail, without a final sigmoid activation, as the target is a continuous temperature field rather than a segmentation mask. The model contains 3,931,047 trainable parameters.

## MeshGraphNet

We evaluate MeshGraphNet [39] as a graph-based neural operator for predicting temperature fields on unstructured surface meshes.

Problem formulation: Each sample is represented as a graph $G = ( V , E )$ constructed from surface mesh nodes. The model learns a mapping from node- and edge-level features to a scalar temperature field:

$$
( \mathbf { X } _ { V } , \mathbf { X } _ { E } , G )  \mathbf { Y } _ { V } ,
$$

where ${ \bf X } _ { V } \in \mathbb { R } ^ { N \times { 4 8 } }$ are node features, $\mathbf { X } _ { E } \in \mathbb { R } ^ { E \times 3 }$ are edge features, and $\mathbf { Y } _ { V } \in \mathbb { R } ^ { N \times 1 }$ is the target field. Taking only surface meshes as an input (the main quantities of interest), the MeshGraphNet is effectively a surface-based model.

Node representation: Each node encodes local geometric and physical information, consisting of:

• spatial coordinates (3D, normalized),

• surface normals (3D),

• surface type encoding (5D one-hot: wall, inlet, outlet, symmetry, body),

• boundary condition encoding (36D heat source representation),

• inflow velocity scalar (1D).

This results in a 48-dimensional node feature vector.

Edge representation: Edges are defined by mesh connectivity, and each edge is associated with a normalized displacement vector:

$$
\mathbf { x } _ { i j } = \mathbf { p } _ { j } - \mathbf { p } _ { i } ,
$$

scaled by the domain diagonal to ensure numerical stability. This encodes local geometric interactions between neighboring surface elements.

Model architecture: MeshGraphNet consists of a node encoder, edge encoder, processor, and decoder, each implemented as 2-layer MLPs with hidden dimension 128. The processor performs 15 message passing steps with sum aggregation. Node and edge inputs are 48- and 3-dimensional respectively, and the output is a scalar per node. The model contains 2,339,201 trainable parameters.

## Transolver

We evaluate Transolver [23] as a transformer-based neural operator for learning mappings between functional inputs and solution fields on unstructured point clouds.

Problem formulation: Each sample consists of an unstructured point cloud representing a 3D surface with N points. The model learns a mapping

$$
( \mathbf { p } , \mathbf { u } )  \mathbf { y } ,
$$

where $\mathbf { p } \in \mathbb { R } ^ { N \times 3 }$ denotes spatial coordinates, $\mathbf { u } \in \mathbb { R } ^ { N \times 4 0 }$ denotes point-wise input features, and $\mathbf { y } \in \mathbb { R } ^ { \tilde { N } \times 1 }$ is the target temperature field. In the context, of our contribution we focus on surface points since these are the main points of interest. Thus, Transolver is effectively a surface model.

Input representation: For each point, the input features consist of:

• surface normals (<sup>R3</sup>),

• geometric coordinates encoded via normalization to $[ 0 , 1 ] ^ { 3 }$

• global boundary-condition features broadcast to all points.

The boundary condition encoding includes up to 6 heat source objects (cuboids or cylinders), each represented by a 6D feature vector encoding position, temperature, and size parameters. These are concatenated and zero-padded to a fixed-length representation, together with a normalized inflow velocity scalar. This results in a 40 dimensional functional input per point.

Model architecture: We use 8 Transolver blocks with hidden dimension 128 and 8 attention heads. The physics token space is discretized into 256 slices, and spatial coordinates are embedded into a 3-dimensional positional encoding. The model contains 1,539,649 trainable parameters.

## Training

All models are trained for 100 epochs on 8 NVIDIA H100 GPUs using distributed data parallel training, with the AdamW optimizer and ReduceLROnPlateau scheduling with a reduction factor of 0.5 and patience of 5 epochs, reducing the learning rate when validation loss stops improving. Mean squared error (MSE) between predicted and ground-truth temperature fields serves as the training objective. Mixed-precision training and gradient clipping are applied for numerical stability and efficiency.

![](images/CoolData-07.jpg)  
Figure 13: Visualization of loss function for all three models trained on a dataset with 50, 000 different designs.

To assess the effect of dataset size on model performance, each model is trained at three scales: 10, 000, 30, 000, and 50, 000 samples, using a fixed 90/10 train/validation split. The scaling is cumulative: the 10, 000 samples used at the first scale are a strict subset of those used at subsequent scales, ensuring that performance differences reflect the addition of new training data rather than changes in the training distribution. Evaluation is performed on a fixed held-out test set of 1, 000 samples not seen during training at any scale.

MeshGraphNet and Transolver operate on variable-length point clouds and surface meshes respectively, precluding standard batching. Instead, the DataLoader prefetches 48 samples to keep the GPU pipeline saturated, while the model processes one sample per forward pass. This prefetching strategy significantly reduces I/O bottlenecks and CPU idle time compared to sequential single-sample loading.

MeshGraphNet and Transolver are implemented via NVIDIA PhysicsNeMo [24], and the 3D U-Net via pytorch-3dunet [40]. Specific hyper-parameters used are shown in Table 3 and an exemplary plot of the loss function is shown in Figure 13.
<table><tr><td>Hyperparameter</td><td>3D U-Net</td><td>MeshGraphNet</td><td>Transolver</td></tr><tr><td>Epochs</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Batch size</td><td>16</td><td>1 (prefetch 48)</td><td>1 (prefetch 48)</td></tr><tr><td>Learning rate</td><td>1 × 10 -4</td><td> $\bar { 1 } \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>LR scheduler</td><td colspan="3">ReduceLROnPlateau (factor=0.5, patience=5)</td></tr><tr><td>Gradient clip</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Loss function</td><td>MSE</td><td>MSE</td><td>MSE</td></tr><tr><td>Mixed precision</td><td>FP16</td><td>FP16</td><td>FP16</td></tr><tr><td>Weight decay</td><td>10⁻2</td><td>10⁻²</td><td>10⁻2</td></tr><tr><td>Model selection</td><td>Best val. MSE</td><td>Best val. MSE</td><td>Best val. MSE</td></tr></table>

Table 3: Training hyperparameters for all models. All models are trained with distributed data parallel (DDP) on 8 NVIDIA H100 GPUs with AdamW optimisation.

## F Evaluation

While all models in this study are trained using a Mean Squared Error (MSE), we use additional quantities to evaluate model performance. The key metrics used in this study are:

Mean Squared Error (MSE):

$$
\mathbf { M S E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( T _ { i } - \hat { T } _ { i } ) ^ { 2 }
$$

Mean Absolute Error (MAE):

$$
\mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| T _ { i } - \hat { T } _ { i } \right|
$$

Maximum Absolute Error (MaxAE):

$$
\mathrm { M a x A E } = \operatorname* { m a x } _ { i \in \{ 1 , n \} } \left| T _ { i } - \hat { T } _ { i } \right|
$$

Coefficient of Determination $( R ^ { 2 } )$ :

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { n } ( T _ { i } - \hat { T } _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { n } ( T _ { i } - \bar { T } _ { i } ) ^ { 2 } }
$$

with $\begin{array} { r } { \bar { T } _ { i } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } T _ { i } } \end{array}$

Since we work on the one hand we benchmark a volumetric model (3D U-Net) and on the other hand with two surface centric models (MeshGraphNet and Transolver), the corresponding evaluation metrics are considered over different sets. For the 3D U-Net we evaluate the error for each voxel, i.e., we transform the true result also into a voxel-based representation an evaluate the differences voxel wise. For MeshGraphNet and Transolver we evaluate the differences across all nodes of the corresponding surface discretizations.

## G Notation and Main Physical Quantities

The dataset provides a comprehensive set of physical quantities commonly used in CFD. These are divided into volume and surface fields, each represented on an unstructured mesh and stored in the CGNS format. Units are provided in SI.

## Volume Field Quantities

Each volume sample contains the following fields:

• Velocity $\vec { v } ( \mathrm { m / s } ) \mathrm { : }$ A 3D vector field representing the fluid velocity components in the x, y, and z directions (Velocity\_0, Velocity\_1, Velocity\_2).

• Pressure $p \left( \mathrm { P a } \right)$ : The static pressure field.

• Temperature $T \ ( \mathrm { K } )$ : The temperature of the fluid.

• TurbulentKineticEnergy $k \ : ( \mathrm { m } ^ { 2 } / \mathrm { s } ^ { 2 } )$ : The specific turbulent kinetic energy.

• TurbulentDissipationRate $\varepsilon \left( \mathrm { m } ^ { 2 } / \mathrm { s } ^ { 3 } \right)$ : The dissipation rate of turbulent kinetic energy.

## Surface Field Quantities

Each surface sample includes wall-related quantities defined on the geometry surface:

• AreaMagnitude $A ( \mathrm m ^ { 2 } )$ : The surface area associated with each wall element.

• HeatTransferCoefficient $h \left( \mathrm { W } / ( \mathrm { m } ^ { 2 } \cdot \mathrm { K } ) \right)$ : The local convective heat transfer coefficient at the wall.

• Normal ⃗n (unitless): A unit 3D vector normal to the surface, with components Normal\_0, Normal\_1, and Normal\_2.

• Temperature $T _ { \mathrm { b o d y } } ( \mathrm { K } )$ : The wall temperature, which is constant across a body.

• Pressure $p \left( \mathrm { P a } \right)$ : The static pressure on the wall surface.

• WallShearStressMagnitude $| \vec { \tau } _ { \mathrm { w } } | \ ( \mathrm { P a } )$ : The magnitude of the wall shear stress vector.

• WallShearStress $\vec { \tau } _ { \mathrm { w } } \quad ( \mathrm { P a } ) \colon \mathrm { ~ \bf ~ A ~ }$ 3D vector representing wall shear stress components in the $x , \ z ,$ and z directions (WallShearStress\_0, WallShearStress\_1, WallShearStress\_2).

## Derived Heat Transfer Related Quantities

Based on the saved volumetric and surface data, appropriate engineering related quantities can be derived, e.g., taking appropriate surface integrals. Below we summarize the heat transfer related quantities we use for evaluation of the dataset:

• Mean HTC $\bar { h } _ { \mathrm { a w } } \ \mathrm { ( W } \mathrm { m } ^ { - 2 } \mathrm { K } ^ { - 1 } \mathrm { ) }$ : Area-weighted mean surface-integral, i.e., $\begin{array} { r l } { \bar { h } _ { \mathrm { a w } } } & { { } = } \end{array}$ $\textstyle \int h \mathrm { d } A / \int \mathrm { d } A$

• Body-to-inlet $\Delta T _ { \mathrm { b o d y } }$ (K): Temperature difference between inflow and body, i.e., $\Delta T _ { \mathrm { b o d y } } =$ $T _ { \mathrm { b o d y } } - T _ { \mathrm { i n } }$

• Body heat dissipation $Q _ { \mathrm { b o d y } } ( \mathrm { W } ) { \mathrm { ; } }$ : Heat flux integrated over a single body, i.e. $Q _ { \mathrm { b o d y } } =$ $\begin{array} { r } { \sum _ { i \in \mathrm { f a c e s } } h _ { i } A _ { i } \dot { \Delta \mathcal { T } } _ { \mathrm { b o d y } } } \end{array}$

• Total heat dissipation $Q _ { \mathrm { t o t a l } } ( \mathrm { W } )$ : Sum of $Q _ { \mathrm { b o d y } }$ over all heated bodies in a sample.

## NeurIPS Paper Checklist

1. Claims Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope? Answer: [Yes] Justification: Described in the Section 1.

2. Limitations Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes] Justification: This is discussed in the Section 6.

3. Theory assumptions and proofs Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof? Answer: [N/A] Justification: The paper does not contain theoretical results.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Justification: Details regarding the models, training, and inference processes are provided in Appendix E.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

## Answer: [Yes]

Justification: The dataset can be accessed at: https://huggingface.co/datasets/ bgce/cooldata-v2. An accessor library is provided and documented at: https:// cooldata.readthedocs.io This library supports loading, visualization, and conversion for various ML pipelines. Furthermore, the scripts for training the models used in this paper are available in the following repository: https://github.com/peteole/flow\_field\_ dataset.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results? Answer: [Yes]

Justification: Further details are discussed in Appendices E and F.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Justification: We do not address extensive statistical evaluations, since the focus is on the dataset. Nevertheles, the distributions for the dataset have been provided.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

## Answer: [Yes]

Justification: The paper provides details on the hardware used for dataset creation and model training. Training and inference times have been provided in the Section 4.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: The research was conducted in accordance with the NeurIPS Code of Ethics.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [Yes]

Justification: The broader societal impact of this work is discussed in Section 1 and Appendix A. While we anticipate predominantly positive impacts, we acknowledge that faster design tools could indirectly accelerate the production of higher-power devices, potentially increasing energy consumption at scale. However, the primary use case of reducing CFD simulation costs during the design phase is expected to yield a net positive effect on energy efficiency.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

## Answer: [N/A]

Justification: To the best of our knowledge, the dataset does not contain any personally identifiable information and poses no potential safety risks.

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

## Answer: [Yes]

Justification: All models and accompanying code used for the experiments have been cited, and their respective licenses have been adhered to.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

## Answer: [Yes]

Justification: Detailed documentation for the accompanying assets is available at: https: //cooldata.readthedocs.io.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

## Answer: [N/A]

Justification: This work involves neither crowdsourcing nor research with human subjects.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: This research did not involve crowdsourcing or studies with human subjects.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

## Answer: [N/A]

Justification: LLM was utilized solely for writing, editing, and formatting purposes, and was not a part of the core methodology.