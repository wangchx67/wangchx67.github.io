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
        },{id: "post-dynamic-region-aware-convolution-drconv",
        
          title: "Dynamic Region-Aware Convolution (DRConv)",
        
        description: "旷世CVPR2021论文Dynamic Region-Aware Convolution介绍及其在底层视觉中的应用",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/dynamic-region-aware-convolution/";
          
        },
      },{id: "post-qwen-image-technique-report-and-code",
        
          title: "qwen-image technique report and code",
        
        description: "学习下qwen-image",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/qwen-image/";
          
        },
      },{id: "post-flux2-0-technique-report-and-code",
        
          title: "flux2.0 technique report and code",
        
        description: "学习下flux2.0",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/flux2.0/";
          
        },
      },{id: "post-different-diffusion-prediction-and-loss",
        
          title: "Different diffusion prediction and loss",
        
        description: "学习噪声/图像/速度预测以及对应的优化方式（读JiT想要汇总记录下）",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2026/different-diffusion-prediction-and-loss/";
          
        },
      },{id: "post-flash-attention",
        
          title: "Flash attention",
        
        description: "学习Flash attention",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/flash-attention/";
          
        },
      },{id: "post-architecture-design-of-diffusion-model",
        
          title: "Architecture design of diffusion model",
        
        description: "目前主流扩散模型中的架构设计",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/architecture-design-of-diffusion-model/";
          
        },
      },{id: "post-baisc-diffusion-theroy",
        
          title: "Baisc diffusion theroy",
        
        description: "推导一下扩散模型",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/blog/2025/basic-diffusion-theroy/";
          
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
