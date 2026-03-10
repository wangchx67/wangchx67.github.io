// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/blog/";
          },
        },{id: "nav-publications",
          title: "publications",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-different-diffusion-prediction-and-loss",
        
          title: "Different diffusion prediction and loss",
        
        description: "学习噪声/图像/速度预测以及对应的优化方式",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/different-diffusion-prediction-and-loss/";
          
        },
      },{id: "post-flux-series-technique-report-and-source-code",
        
          title: "flux series - technique report and source code",
        
        description: "学习下flux",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/flux-series-technique-report-and-source-code/";
          
        },
      },{id: "post-flash-attention",
        
          title: "flash attention",
        
        description: "深入学习Flash Attention算法，包含官方源码分析与手撕简单实现",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/flash-attention/";
          
        },
      },{id: "post-architecture-design-of-diffusion-model",
        
          title: "Architecture design of diffusion model",
        
        description: "DiT DDT MMDiT...",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/diffusion-transformer/";
          
        },
      },{id: "post-rope",
        
          title: "ROPE",
        
        description: "旋转位置编码",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/ROPE/";
          
        },
      },{id: "post-baisc-diffusion-theroy",
        
          title: "Baisc diffusion theroy",
        
        description: "推导一下扩散模型",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/basic-diffusion-theroy/";
          
        },
      },{id: "post-noise-model-in-isp",
        
          title: "noise model in ISP",
        
        description: "相机raw的噪声来源和建模",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/noise-model-in-ISP/";
          
        },
      },{id: "post-lliepaper-low-light-image-enhancement-via-structure-modeling-and-guidance",
        
          title: "LLIEpaper-Low-Light Image Enhancement via Structure Modeling and Guidance",
        
        description: "CVPR2023低光照增强论文：基于结构建模和引导的图像增强方法",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/LLIEpaper-structure-modeling-and-guidance/";
          
        },
      },{id: "post-lliepaper-regularizer-free-retinex-decomposition-rfr",
        
          title: "LLIEpaper-Regularizer-Free Retinex Decomposition (RFR)",
        
        description: "CVPR2023低光照增强论文：不使用额外先验和正则项的Retinex分解方法",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/LLIEpaper-regularizer-free-retinex-decomposition/";
          
        },
      },{id: "post-dynamic-region-aware-convolution-drconv",
        
          title: "Dynamic Region-Aware Convolution (DRConv)",
        
        description: "旷世CVPR2021论文Dynamic Region-Aware Convolution介绍及其在底层视觉中的应用",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2023/dynamic-region-aware-convolution/";
          
        },
      },{id: "post-流模型生成模型-flow-based-generative-model",
        
          title: "流模型生成模型 (Flow-based Generative Model)",
        
        description: "深入理解基于流的生成模型及其在底层视觉任务中的应用",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2022/flow-based-generative-model/";
          
        },
      },{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%77%61%6E%67%63%68%65%6E%78%69_%64%61%69%6C%79@%31%36%33.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=339f9gwAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/wangchx67/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
