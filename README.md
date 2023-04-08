# dataset_gradcam_plus_plus

For bigger dataset (>5000) , we recommand to split the dataset into numerous batchs of 5000 samples.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-library">About The Library</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
<!-- ABOUT THE LIBRARY -->
## About The Library
As we know, Grad-CAM ++ Algorithms which gives a visual Explainations for the decisions made by Deep Learning Model. Despite its advantages, the method have drawbacks.
* GRAD-CAM ++ was designed at the first for time series
* The visual explanation can only be used for local proediction, not a a dataset level.

Our library name dataset_gradcam_plus_plus, not only implement Grad-CAM++ at a local level for 1d signal/Time series, but also enable the user, to get feature importance for classification at a dataset level.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][Python]][Python-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
