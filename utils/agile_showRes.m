close all

%% Save figures?
save_flag = false;

%% Get resutls
Wg = para.W;
Hg = para.H;
W  = STATS.W;
H  = STATS.H;
Fval = STATS.Fval;

%% Remove the bias dimensionalities from W and H
vecD = para.vecD + 1;
numV = length(vecD);
id_start = 0;
idx_remove = [];
for v = 1 : numV
    id_tmp = id_start + vecD(v);
    idx_remove = cat(1,idx_remove,id_tmp);
    id_start = id_tmp;
end
W(idx_remove,:) = [];
H(idx_remove,:) = [];

%% Adjust the values of matrices for visualization
W_img  = myAdjust(abs(W),0,255);
H_img  = myAdjust(abs(H),0,255);
T_img  = myAdjust(W_img+H_img,0,255);
Wg_img = myAdjust(abs(Wg),0,255);
Hg_img = myAdjust(abs(Hg),0,255);
Tg_img = myAdjust(Wg_img+Hg_img,0,255);

%% Set the label size in figures
font_size = 16;
tick_size = 10;

%% Figure 1: designed weight matrices
fig1 = figure;
subplot(3,1,1);
image(Wg_img')
a = get(gca,'xticklabel');
set(gca,'xticklabel',a,'fontsize',tick_size);
ylabel('$\mathbf{W}^{\ast\top}$','interpreter','latex','fontsize',font_size);
subplot(3,1,2);
image(Hg_img')
a = get(gca,'xticklabel');
set(gca,'xticklabel',a,'fontsize',tick_size);
ylabel('$\mathbf{H}^{\ast\top}$','interpreter','latex','fontsize',font_size);
subplot(3,1,3);
image(Tg_img')
colormap(gray)
a = get(gca,'xticklabel');
set(gca,'xticklabel',a,'fontsize',tick_size);
ylabel('$\boldmath\Theta^{\ast\top}$','interpreter','latex','fontsize',font_size);
suptitle('Designed weight matrices')

%% Figure 2: learned weight matrices
fig2 = figure;
subplot(3,1,1);
image(W_img')
a = get(gca,'xticklabel');
set(gca,'xticklabel',a,'fontsize',tick_size);
ylabel('$\mathbf{W}^\top$','interpreter','latex','fontsize',font_size);
subplot(3,1,2);
image(H_img')
a = get(gca,'xticklabel');
set(gca,'xticklabel',a,'fontsize',tick_size);
ylabel('$\mathbf{H}^\top$','interpreter','latex','fontsize',font_size);
subplot(3,1,3);
image(T_img')
colormap(gray)
a = get(gca,'xticklabel');
set(gca,'xticklabel',a,'fontsize',tick_size);
ylabel('$\boldmath\Theta^\top$','interpreter','latex','fontsize',font_size);
suptitle('Weight matrices learned by AGILE')

%% Figure 3: Convergence analysis 
fig3 = figure;
style_set = {'r-o','g-x','b-s','c-d','k-*'};
for val_id = 1 : size(Fval,1)
    plot(log(Fval(val_id,:)),style_set{val_id});
    hold on
end
legend('Loss','R1','R2','R3','Obj');
xlabel('Number of iterations','fontsize',font_size)
ylabel('Objective value (log-scale)','fontsize',font_size)
    
%% Save the figures 
if save_flag
    saveas(fig1,'agile_demo_designed','png');
    saveas(fig2,'agile_demo_learned','png');  
    saveas(fig3,'agile_demo_converge','png');        
end
