<br>

## <div align="center">**MentalMR: Human-Centered Framework for Empowering Psychological Understanding**</div>

<details open>
<summary>Introduction</summary>
  
As applications of ML algorithms in real life become increasingly common, ethical issues of AI solutions emerge especially in a sensitive human-related field such as the medical care industry. The output of the responsible AI framework proposed are images demonstrated to be in both high accuracy and high interpretability in field experiments with accuracy higher than 90% by prediction based on bare human eyes. More importantly, the images generated under the responsible framework enable experts in the field of mental health to diagnose mental disorders even if they do not know ML algorithms.
  
**In this branch, VAE + t-SNE is used to see if the pattern of the images are explicitly identified and thus proving the quality of the images made.**

<details open>
<summary>Responsibility of the research</summary>
  
**Tutun Salih:** Project owner. Providing and gathering original data; leading and indicating the design of the research.
  
**Gorden Li(Kehao Li)**: Head research assistant, responsible of implementing alogirithms and writing codes to achieve the design of Professor Tutun Salih.
  
**Yuxiang Wu**: Research assistant, assist head research assistant.

</details>
  
</details>

<details open>
<summary>Result</summary>

- Images are represented in a latent space based on VAE. After getting the high-dimension latent expression of an image, Gaussian Mixture Model is used to identify clusters. Meanwhile, t-SNE is applied to μ in latent expression to visualize images pattern on a 2D plane.
  
  Latent-expressing images after applying t-SNE:
  
  ![VAE+T-SNE](https://user-images.githubusercontent.com/72702872/169719438-2d6406b4-a53e-4954-b469-401cc0d4fa1d.png)
  
  Latent-expressing images after applying t-SNE, meanwhile using GMM to find out clusters according to latent expression, red = positive case, blue = negative case. We can see there is a clear difference between positive and negative samples:
  
  ![VAE classification](https://user-images.githubusercontent.com/72702872/169719488-9343fe4e-9471-4cf6-b2b9-9ac100c91736.png)
  
  Pattern Tree is as follows, in different location of 2D plane, there appear to be different in place showing darker pixels:
  
  ![image](https://user-images.githubusercontent.com/72702872/169720167-2b5c8333-209e-4259-ac46-70ef8c934f20.png)
  
  Here's an example of positive image and negative:
  
  ![image](https://user-images.githubusercontent.com/72702872/169719533-fdadc724-9188-4013-a6ad-ce16b4b9a889.png) 

</details>

</details>

</details>

## <div align="center">Contact</div>

email: likehao1006@gmail.com

LinkedIn: https://www.linkedin.com/in/kehao-li-06a9a2235/

ResearchGate: https://www.researchgate.net/profile/Gorden-Li

<br>

