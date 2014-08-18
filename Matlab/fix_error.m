% moves some Matlab costum libraries out of the way
% written by Stefan Leutenegger, 8/2011

% determine the build and mex directory
c='';
directory='';
switch computer
    case 'GLNXA64' 
        c='glnxa64';
        directory='unix64';
    case 'GLNX86' 
        c='glnx86';
        directory='unix32';
    case 'PCWIN' 
        c='pcwin';
        directory='win32';
    case 'PCWIN64' 
        c='pcwin64';
    case 'MACI' 
        c='maci';
        directory='apple';
    case 'MACI64' 
        c='maci64';
    otherwise
        disp('error determining architecture');
end

 % kind of dangerous, but a nice hack:
current=cd;

cd([matlabroot '/sys/os/' c]);
% find out if we need to move away things
listing1=dir('libstdc++*');
listing2=dir('libgcc_s*');
listsize=size(listing1)+size(listing2);
if(listsize(1)~=0)
    system(['echo I will create a folder named "moved_away" in ' matlabroot '/sys/os/' c ' to place some conflicting libraries. You will be prompted for the sudo password:']);
    if ~isdir('moved_away')
        system('sudo mkdir moved_away');
    end
    system('sudo mv libstdc++* moved_away');
    system('sudo mv libgcc_s* moved_away');
    system('echo You will have to restart matlab before you can use brisk. I am sorry.');
end
cd(current);