function figurestyle(fontsize)
% Apply consistent style to MATLAB axes/figures.
    ax = gca; 
    ax.FontSize = fontsize;
    ax.TickLabelInterpreter = 'latex';
    if ~isempty(ax.XLabel)
        ax.XLabel.Interpreter = 'latex';
    end
    if ~isempty(ax.YLabel)
        ax.YLabel.Interpreter = 'latex';
    end
    if ~isempty(ax.ZLabel)
        ax.ZLabel.Interpreter = 'latex';
    end
    grid(ax, 'on');
    ax.XMinorGrid = 'off'; 
    ax.YMinorGrid = 'off'; 
    lines = findall(ax, 'Type', 'Line');
    for i = 1:numel(lines)
        lines(i).LineWidth = 2.5;
    end
    leg = findobj(ax.Parent, 'Type', 'Legend');
    for i = 1:numel(leg)
        leg(i).Interpreter = 'latex';
        leg(i).FontSize = fontsize;
        leg(i).Box = 'off';
    end
    cbar = findall(ax.Parent, 'Type', 'ColorBar');
    for i = 1:numel(cbar)
        cbar(i).TickLabelInterpreter = 'latex';
        cbar(i).FontSize = fontsize;
    end
end