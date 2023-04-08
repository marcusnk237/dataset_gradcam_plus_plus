# dataset_gradcam_plus_plus
The first implementation of Grad-CAM ++ at a dataset-level.

As we know, Grad-CAM ++ Algorithms which gives a visual Explainations for the decisions made by Deep Learning Model. 
However, the explanation is give at local level (for a sample). Our library , not only implement Grad-CAM++ at a local level for 1d signal/Time series, but also give a dataset level explanability. 
For bigger dataset (>5000) , we recommand to split the dataset into numerous batchs of 5000 samples.

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-library">About The Project</a>
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

