## Observation of Jones-Roberts solitons in a paraxial quantum fluid of light

### Codes:
- contrast.py contains all image density and phase extraction from raw data
- velocity.py to extract information from fields array
- data_processing.py to run the treatement

### Description of the Dataset
Download the data on Zenodo: link

This dataset contains numerical and experimental data used to generate the figures in our article. The provided files include raw interferograms, processed field data, and computed quantities such as energy spectra and velocity fields.

#### **Figures and Data Sources**

-   **Fig1**: Data from `07191910`
-   **Fig2**: Data from `09061965`
-   **Fig3**: Data from `09101636`
-   **Fig4**: Data from `spectral_analysis`

#### **Field Data**

The `field.npy` files contain 4D NumPy arrays with shape `(i, j, Ny, Nx)`, where:

-   `i` represents the number of time steps,
    
-   `j` corresponds to the number of images averaged at each time step,
    
-   `Ny, Nx` are the spatial dimensions.
    
-   `field_ref.npy` corresponds to a reference Gaussian field without vortices.
    
-   `field_vortex.npy` contains fluid data with a single vortex, used to measure vortex size at each time step.
    

#### **Computed Quantities**

The dataset also includes various derived quantities, such as energy distributions and velocity fields.

#### **Spectral Analysis and Interferograms**

An exception is the dataset related to **Figure 4**, which contains `.tiff` images representing raw interferograms. The field must be reconstructed from these raw data. The folder also includes correlation curves and energy spectra.

This dataset is openly available for further analysis and verification.
