# C vs D: paired comparison

CER difference = CER(D) − CER(C); negative favours D. 95% CI from a paired bootstrap (1000 resamples, seed 42). Exact match = normalised exact match (NEM).

| dataset | n | CER C | CER D | CER(D)−CER(C) | 95% CI | outputs differ | exact only D | exact only C |
|---|---|---|---|---|---|---|---|---|
| s2l_test500 | 500 | 0.3592 | 0.3724 | +0.0132 | [-0.0069, +0.0332] | 101 | 12 | 9 |
| mathspeech | 1101 | 0.3441 | 0.3237 | -0.0204 | [-0.0461, -0.0046] | 88 | 22 | 8 |

## s2l_test500: exact only with D (12 clips)

| clip | Whisper best-1 | reference | C | D |
|---|---|---|---|---|
| audio/365_spk_39.wav | 6 text take away, 3y equals 12. | `6x - 3y = 12` | `6 - 3y = 12` | `6x - 3y = 12` |
| audio/1390_spk_40.wav | See sub 2 open parenthesis e close parenthesis equals 2. | `C_ { 2 } ( E ) = 2` | `S_{2}(e)=2` | `C_2(E)=2` |
| audio/2244_spk_17.wav | f equals sine 3x. | `f(x) = \sin 3x` | `f=\sin 3x` | `f(x)=\sin 3x` |
| audio/2244_spk_39.wav | f of x equals sending 3x now. | `f(x) = \sin 3x` | `f(x)=	ext{sin}(3x)` | `f(x)=\sin 3x` |
| audio/2886_spk_12.wav | Gamma sub one comma two equals two p, rho, t sub one comma two squared. | `\Gamma_ { 1,2 } =2\pi\rho t_ { 1,2 } ^2` | `\Gamma_{1,2} = 2 p \rho t_{1,2}^2` | `\Gamma_{1,2} = 2 \pi \rho t_{1,2}^2` |
| audio/3293_spk_40.wav | Kappa equals 1, 2. | `\kappa =1,2` | `\kappa = 1.2` | `\kappa = 1, 2` |
| audio/6076_spk_12.wav | The Bernoulli's equation equals pressure plus one-half density velocity squared plus density gravity height equals constant | `P + \frac{1}{2} \rho v^2 + \rho g h = \text{constant}` | `P + \frac{1}{2} \rho v^2 + \rho g h = C` | `P + \frac{1}{2} \rho v^2 + \rho g h = \text{constant}` |
| audio/6076_spk_9.wav | the Bernoulli's equation equals pressure plus one half density velocity squared plus density gravity height equals constant | `P + \frac{1}{2} \rho v^2 + \rho g h = \text{constant}` | `P + \frac{1}{2} \rho v^2 + \rho g h = C` | `P + \frac{1}{2} \rho v^2 + \rho g h = \text{constant}` |
| audio/6263_spk_14.wav | The density of states for a 3D3 particle is Ge equals 2m over h-bar squared to the 3.5 times squared E. | `g(E) = \left(\frac{2 m}{\hbar^2}\right)^{3/2} \sqrt{E}` | `g(E) = \left( \frac{2m}{\hbar^2} \right)^{3/2} E^{1/2}` | `g(E) = \left( \frac{2m}{\hbar^2} \right)^{3/2} \sqrt{E}` |
| audio/8549_spk_17.wav | The bar of zero equals alpha. | `\bar { v } ( 0 ) = \alpha` | `\overline{0} = \alpha` | `\bar{v}(0)=\alpha` |
| audio/8818_spk_9.wav | WSUP2 | `W_2` | `W^{2}` | `W_2` |
| audio/9086_spk_4.wav | x squared plus 2i x minus 4 equals 0. Solving for x gives x equals i plus or minus square root of 5. | `x^2 + 2ix - 4 = 0 \quad \Rightarrow \quad x = i \pm \sqrt{5}` | `x^2 + 2ix - 4 = 0` | `x^2 + 2ix - 4 = 0 \quad \Rightarrow \quad x = i \pm \sqrt{5}` |

## s2l_test500: exact only with C (9 clips)

| clip | Whisper best-1 | reference | C | D |
|---|---|---|---|---|
| audio/517_spk_40.wav | A is equal to 2xy. | `A = 2xy` | `A=2xy` | `i=2xy` |
| audio/1957_spk_39.wav | Epsilon sub J equals zero that I have a bandage in here. | `\varepsilon_j=0` | `\varepsilon_j=0` | `\varepsilon_j=0 \quad \text{that I have a bandage in the } \mathbb{R}` |
| audio/3050_spk_13.wav | head lambda sub n of v equals lambda of v plus little o sub p of one | `\hat { \lambda } _n ( v ) = \lambda ( v ) + o_p ( 1 )` | `\hat{\lambda}_n (v) = \lambda (v) + o_p (1)` | `\hat{\lambda}_n (v) = \lambda (v) + o_v (1)` |
| audio/3922_spk_17.wav | m equals zero comma dot dot dot comma capital N | `m=0 , \dots , N` | `m=0,\ldots,N` | `m=0,\ldots\,\text{capital }N` |
| audio/5671_spk_40.wav | Slope of 7 divided by 16. | `\frac{7}{16}` | `\frac{7}{16}` | `7 \div 16` |
| audio/6051_spk_5.wav | The area of a triangle is one half times the base times the height. | `A = \frac{1}{2}bh` | `A = \frac{1}{2} b h` | `\text{Area} = \frac{1}{2} \times \text{base} \times \text{height}` |
| audio/7698_spk_40.wav | The sample variance equals the sum of x of i minus the sample mean squared divided by n minus 1. | `s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})^2}{n-1}` | `s^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n - 1}` | `s^2 = \frac{\sum (x_i - \bar{x})^2}{n - 1}` |
| audio/8456_spk_7.wav | u of xi sub n. | `u ( \xi_n )` | `u(\xi_n)` | `u(\psi_n)` |
| audio/10104_spk_40.wav | The divergence of a second-order tensor equals partial t x x with respect to x plus partial t y with respect to y. | `\nabla \cdot \mathbf{T} = \frac{\partial T_{xx}}{\partial x} + \frac{\partial T_{yy}}{\partial y}` | `\nabla \cdot \mathbf{T} = \frac{\partial T_{xx}}{\partial x} + \frac{\partial T_{yy}}{\partial y}` | `\operatorname{div} T = \frac{\partial T_{xx}}{\partial x} + \frac{\partial T_{yx}}{\partial y}` |

## mathspeech: exact only with D (22 clips)

| clip | Whisper best-1 | reference | C | D |
|---|---|---|---|---|
| audio/0039.mp3 | Cosine x cosine x is cosine squared x. | `$\cos x \cos x = \cos^{2}x$` | `\cos x \cdot \cos x = \cos^2 x` | `\cos x \cos x = \cos^2 x` |
| audio/0041.mp3 | d dy dx times dx dw times dw d theta. | `$ \frac{dy}{dx} \times \frac{dx}{dw} \times \frac{dw}{d\theta} $` | `\frac{d}{dy} \frac{dx}{dx} \frac{dw}{d\theta} \frac{dw}{d\theta}` | `\frac{dy}{dx} \times \frac{dx}{dw} \times \frac{dw}{d\theta}` |
| audio/0095.mp3 | And x to the n minus 1. | `$nx^{n-1}$` | `x^{n-1}` | `n x^{n-1}` |
| audio/0144.mp3 | I want to have r squared omega squared. | `$\frac{1}{2}mr^{2}\omega^{2}$` | `r^2 \omega^2` | `\frac{1}{2} m r^{2} \omega^{2}` |
| audio/0162.mp3 | minus theta cubed over six. | `$\theta-\frac{\theta^{3}}{6}$` | `-\frac{\theta^3}{6}` | `\theta - \frac{\theta^3}{6}` |
| audio/0206.mp3 | free y plus z equals zero. | `$x+3y+z=0$` | `f(y) + z = 0` | `x + 3y + z = 0` |
| audio/0212.mp3 | d by dt of r dot r is zero. | `$\displaystyle \frac{d}{dt}(\mathbf{r}\cdot\mathbf{r})=0$` | `\frac{d}{dt} ( r \cdot r ) = 0` | `\frac{d}{dt} ( \mathbf{r} \cdot \mathbf{r} ) = 0` |
| audio/0400.mp3 | f of x equals g of x plus a constant. | `$f(x)=g(x)+\text{constant}$` | `f(x) = g(x) + c` | `f(x) = g(x) + \text{constant}` |
| audio/0402.mp3 | It's 124 x to the fourth plus 2 to the power 6 plus c. | `$\frac{1}{24}x^{4}+2^{6}+c$` | `124 x^4 + 2^6 + c` | `\frac{1}{24} x^{4} + 2^{6} + c` |
| audio/0438.mp3 | Minus the integral from b to a of f dx. | `$\displaystyle -\int_{b}^{a}f(x)dx$` | `-\int_a^b f \, dx` | `-\int_b^a f(x)\,dx` |
| audio/0544.mp3 | The integral of dx over x squared times the square root of. | `$\int\frac{dx}{x^{2}\sqrt{1+x^{2}}}$` | `\int \frac{dx}{x^2 \sqrt{x}}` | `\int \frac{dx}{x^2 \sqrt{1+x^2}}` |
| audio/0567.mp3 | b over squared. | `$\frac{b}{(x-1)^{2}}$` | `\frac{b}{squared}` | `\frac{b}{(x-1)^2}` |
| audio/0610.mp3 | The integral from 0 to a of square root of dx. | `$\int_{0}^{a}\sqrt{1+4x^{2}}dx$` | `\int_0^a \sqrt{x} \, dx` | `\int_0^a \sqrt{1+4x^2} \, dx` |
| audio/0662.mp3 | A half x minus a half tan inverse x plus x. | `$\frac{1}{2}x-\frac{1}{2}\tan^{-1}x$` | `\frac{1}{2}x - \frac{1}{2} \tan^{-1} x + x` | `\frac{1}{2}x - \frac{1}{2} \tan^{-1} x` |
| audio/0836.mp3 | negative four plus lambda squared plus lambda squared plus | `$-4+\lambda^{2}$` | `-4 + \lambda^2 + \lambda^2` | `-4 + \lambda^2` |
| audio/0866.mp3 | square root of will be cosine theta. | `$\sqrt{1-x^{2}}=\cos\theta$` | `\sqrt{\cos \theta}` | `\sqrt{1 - x^2} = \cos \theta` |
| audio/0905.mp3 | integral of x dx y dy. | `$\int x\,dx+y\,dy$` | `\int x \, dx \, y \, dy` | `\int x \, dx + y \, dy` |
| audio/0915.mp3 | squared plus 4x squared. | `$3y^{2}+4x^{2}$` | `x^2 + 4x^2` | `3y^2 + 4x^2` |
| audio/0940.mp3 | squared plus z squared is less than one. | `$x^{2}+y^{2}+z^{2}<1$` | `x^2 + z^2 < 1` | `x^2 + y^2 + z^2 < 1` |
| audio/1044.mp3 | x dot of 0 equals to 0. | `$\dot{x}(0)={0}$` | `x(0)=0` | `\dot{x}(0)=0` |
| audio/1053.mp3 | Exponential i omega t. | `$\exp(i\omega t)$` | `e^{i \omega t}` | `\exp(i \omega t)` |
| audio/1069.mp3 | divided by one minus three i. | `$\frac{2+i}{1-3i}$` | `\frac{1}{1-3i}` | `\frac{2+i}{1-3i}` |

## mathspeech: exact only with C (8 clips)

| clip | Whisper best-1 | reference | C | D |
|---|---|---|---|---|
| audio/0019.mp3 | e to the minus b squared times e to the minus x squared. | `$e^{-b^2} e^{-x^2}$` | `e^{-b^2} e^{-x^2}` | `e^{-v^2} e^{-x^2}` |
| audio/0142.mp3 | going r to the 4 over 4. | ` $\frac{r^4}{4}$` | `\frac{r^4}{4}` | `\frac{r}{4} / 4` |
| audio/0328.mp3 | x is equal to 2 to the one-third v to the one-third. | `$x=2^{\frac{1}{3}}v^{\frac{1}{3}}$` | `x=2^{\frac{1}{3}} v^{\frac{1}{3}}` | `x=2^{1/3} v^{1/3}` |
| audio/0599.mp3 | The integral from a to b of the square root of 1 plus dy dx squared dx. | `$\int_{a}^{b}\sqrt{1 +(\frac{dy}{dx})^{2}}dx$` | `\int_a^b \sqrt{1 + \left(\frac{dy}{dx}\right)^2} \, dx` | `\int_a^b \sqrt{x^2} \, dx` |
| audio/0818.mp3 | a1x plus a2y plus a phase z equals c. | `$a_1x+a_2y+a_3z=c$` | `a_1 x + a_2 y + a_3 z = c` | `a_1 x + a_2 y + a \phi e = c` |
| audio/0845.mp3 | d z equals minus one over six times four dx plus dy. | `$dz=-\frac{1}{6}(4dx+dy)$` | `dz = -\frac{1}{6} (4 \, dx + dy)` | `dx = -\frac{1}{6} \times 4 \, dx + dy` |
| audio/0876.mp3 | pi a to the four over two. | `$\frac{\pi a^{4}}{2}$` | `\frac{\pi a^4}{2}` | `\frac{\frac{\pi 8}{4}}{2}` |
| audio/1094.mp3 | So this is at least one half plus 1 over 2n. | `$\frac{1}{2}+\frac{1}{2n}$` | `\frac{1}{2} + \frac{1}{2n}` | `\text{at least } \frac{1}{2} + \frac{1}{2n}` |
