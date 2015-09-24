set(0, 'defaultfigurevisible', 'off');

FRAMES = [1:600*6];

for frm = FRAMES
    load(sprintf('sph-%04d.mat', frm));
    [r,c] = size(particles);
    scatter3(particles(1,:), particles(2,:), particles(3,:), 3, 'filled');
    axis([0 50 0 50 0 50]);
    grid('off');
    box('on');
    title(sprintf('%d particles @ frame #%d', c, frm));
    print(sprintf('sph-%04d.png', frm));
end
