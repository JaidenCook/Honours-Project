import numpy as np
from astropy.io import fits
from astropy.table import Table,Column
import subprocess
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import sys

#Functions:
def subprocess_cmd(command,silent=False):
	process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
	proc_stdout = process.communicate()[0].strip()
	if (silent==False):
		print proc_stdout
	return (proc_stdout)

def find(arr, val): 
	"""This function finds the value in an array closest to some input value val, and returns the index of that value in the original array."""
	index = np.abs(arr - val).argmin()
	return index

def Fiterr(fitfunc,errfunc,init,X,Y,Yerr):
	"""This function takes an input fitting lambda function, and an input lambda error function as well as their associated X and Y, e_Y arrays. It then applies the scipy.leastsq function to find the optimal fitting parameters, running through 10 iterations to refine the fitting parameters. It then returns the fitting parameters and their associated errors.
	"""
	for j in range(10):
		if j ==0:
			out = leastsq(errfunc, init, args=(X,Y,Yerr),full_output=1, epsfcn=0.0001)
			c1 = out[0]
		else:
			out = leastsq(errfunc, c1, args=(X,Y,Yerr),full_output=1, epsfcn=0.0001)
			c1 = out[0]
			cov = out[1]
			c1_err = []
	for i in range(len(c1)):
		c1_err.append(np.sqrt(cov[i,i]))
	
	return c1, c1_err

def Source_fluxes(Nu,Flux,e_Flux,Source):
	"""This function takes an input source (which is of astropy.Table format), and input frequency and flux/error flux lists, and returns the log10 lists of the corresponding numerical flux values for this source."""

	#Initiallising the temporary arrays.
	temp_nu = []
	temp_flux = []
	temp_e_flux = []
	for i in range(len(Flux)):
		if np.isnan(Source[Flux[i]]) == False and Source[Flux[i]] > 0:
			temp_nu.append(Nu[i])
			temp_flux.append(Source[Flux[i]])
			try:
			#In the event there is a flux with no corresponding error in the flux, this will estimate the error as 10% of the corresponding flux.				
				if np.isnan(Source[e_Flux[i]]) == False:	
					temp_e_flux.append(Source[e_Flux[i]])
				elif np.isnan(Source[e_Flux[i]]) == True or Source[e_Flux[i]] < 0:
					temp_e_flux.append(0.1*Source[Flux[i]])
			except KeyError:
				z = 0#Some reason cant leave this blank.

	temp_e_logS = np.array(temp_e_flux)/(np.log(10)*np.array(temp_flux))
	temp_logS = np.log10(np.array(temp_flux))
	temp_logNu = np.log10(np.array(temp_nu)*10e+5)

	return temp_logNu, temp_logS, temp_e_logS

#It might be possible to convert a 2D array into a list using list(). This will save precious memory resources.
def List_Split(Arr1D,Arr2D):
	"""This function takes a 1D and a 2D array as input. It then assigns each column from both arrays as elements to a list, and then outputs that list. The purpose of this function is to create lists that have the necessary format to create astropy tables.
	"""
	Temp_List = []	
	Temp_List.append(list(Arr1D))
	for i in range(len(Arr2D[0,:])):
		Temp_List.append(list(Arr2D[:,i]))
	
	return Temp_List

#########################################################################################################################################
def Main():
	#This section of code creates a new directory for storing the fitted output files.
	subprocess_cmd('mkdir ./Fit_{0}'.format(sys.argv[1]))
	subprocess_cmd('mkdir ./Fit_{0}/plots'.format(sys.argv[1]))

	#Defining Output and error files:
	err_file = open('./Fit_{0}/{0}_e.txt'.format(sys.argv[1]),'w')#Use try statements for this file.
	out_file = open('./Fit_{0}/{0}_o.txt'.format(sys.argv[1]),'w')#write to this instead of using print statements

	#This code is for opening fits files:
	header = fits.getheader("{0}.fits".format(sys.argv[1]))
	data = fits.getdata("{0}.fits".format(sys.argv[1]))
	
	#Table formats the fits file into a user readable output, I will investigate the usefulness of this for my purposes.
	t_sample = Table(data)#Puts fits data into a readable table format.
	col_names = t_sample.colnames #This piece of code saves the table column names as a list.

	#Creating a random list for sample plots:
	rand_sample_list = np.random.randint(0,len(t_sample),size=25)

	#This tuple contains the names of the columns for the new table.
	Cols= ('Name','RA','DEC','l_S300','e_l_S300','q_S300',\
'e_q_S300','l_Chisqd','q_Chisqd','a1','b1','e_a1','e_b1','a2','b2','c2','e_a2','e_b2','e_c2','flag')

	#These two arrays will store source names as well as output, they will eventually be combined into
	#an astropy Table.
	Name_List = np.chararray((len(t_sample)),itemsize=15)
	Out_List = np.zeros((len(t_sample),(len(Cols)-1)))

	#Loading in the frequency values, and the flux/e_flux string names for indexing.
	Nu,Flux,e_Flux = np.loadtxt("Flux_Nu_e_Flux.txt",dtype="f8,S5,S7",usecols=(0,1,2),unpack=True)

	#In this section the quadratic and linear models will be fitted to the data.
	Lin_mod_e = lambda a,x,Yerr: (a[0]*x + a[1])/Yerr
	Lin_mod = lambda a,x: (a[0]*x + a[1])
	Quad_mod_e = lambda a,x,Yerr: (a[0]*(x**2) + a[1]*x + a[2])/Yerr
	Quad_mod = lambda a,x: (a[0]*(x**2) + a[1]*x + a[2])

	#These two functions calculate the Chi squared values for the linear and quadratic fit.	
	Chisqd_Lin = lambda a,x,Y,Yerr: (Y/Yerr-Lin_mod_e(a,x,Yerr))**2
	Chisqd_Quad = lambda a,x,Y,Yerr: (Y/Yerr-Quad_mod_e(a,x,Yerr))**2

	#These two functions calculate the log error in the estimated 300MHz flux density.
	log_lin_e_S = lambda p_e: np.log10(300*10e+5)*p_e[0]#Found using the calculus approach.
	log_quad_e_S = lambda p_e: np.sqrt((np.log10(300*10e+5)**2 *p_e[0])**2 + (np.log10(300*10e+5) * p_e[1])**2)
#########################################################################################################################################
	"""This subroutine runs through the input fits table. The table is read as an astropy table format, where each line is an individual 
source, each source has a varying number of flux data points, so fitting has to be done on a case by case basis. The if statements in this 
case acts as a switch for the more general cases."""
#########################################################################################################################################
	k = len(t_sample)
	for i in range(k):
		try:#Try statement for error checks.
			logNu,logS,e_logS = Source_fluxes(Nu,Flux,e_Flux,t_sample[i])	
			out_file.write("#{0} Source: {1}, \t number of data points = {2}".format(i,t_sample["Name"][i],len(logNu)))
			#Initial linear and quadratic parameter guesses:
			a_lin = [1,1]
			a_quad = [1,1,1]
	
			flag = 'default'#Default flag, set for linear fit.
			Red_Chisqd_l = 0#Setting these as 0 for initial output.
			Red_Chisqd_q = 0
	
			if len(logNu) <= 2:
			#This is the case where the source has no flux values whatsoever.			
				flag = 0
				#Assigning data values to the output tables.
				Name_List[i] = t_sample[i]['Name']
				val_list = [t_sample[i]['RA'],t_sample[i]['DEC'],0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,flag]
	
				for v in range(len(val_list[2:])):#Formatting data
					val_list[2+v] = round(val_list[2+v],5)
			
				#Here I am assigning the relevant values to the output array Out_List.
				for l in range(len(val_list)):
					Out_List[i][l] = val_list[l]		

			elif len(logNu) == 3:
			#Only use linear fitting.
				#Calling the fitting function Fiterr
				C_lin,C_e_lin = Fiterr(Lin_mod_e,Chisqd_Lin,a_lin,logNu,logS,e_logS)
	
				#Calculating the Chi squared values for the linear and quadratic fits.
				Chi2_lin = np.sum(Chisqd_Lin(C_lin,logNu,logS,e_logS))

				#Calculating the reduced Chir squared values for the linear and quadratic fits.
				Red_Chisqd_l = round(Chi2_lin/(len(logNu)-len(C_lin)),3)
				flag = 1#Case for only linear fit with 3 points.

				#Calculating the 300Mhz flux with associated errors.
				lin_S300 = 10**(Lin_mod(C_lin,np.log10(300*10e+5)))
				e_lin_S300 = lin_S300*log_lin_e_S(C_e_lin)*np.log10(10)

				#Assigning data values to the output tables.
				Name_List[i] = t_sample[i]['Name']
				val_list = [t_sample[i]['RA'],t_sample[i]['DEC'],l_S300,e_l_S300,0,0,Red_Chisqd_l,\
0,C_lin[0],C_lin[1],C_e_lin[0],C_e_lin[1],0,0,0,0,0,0,flag]

				for v in range(len(val_list[2:])):#Formatting data
					val_list[2+v] = round(val_list[2+v],5)
				
				#Here I am assigning the relevant values to the output array Out_List.
				for l in range(len(val_list)):
					Out_List[i][l] = val_list[l]

			elif len(logNu) > 3:
			#Use both linear and quadratic fitting.	
				#Calling the fitting function Fiterr
				C_lin,C_e_lin = Fiterr(Lin_mod_e,Chisqd_Lin,a_lin,logNu,logS,e_logS)
				C_quad,C_e_quad = Fiterr(Quad_mod_e,Chisqd_Quad,a_quad,logNu,logS,e_logS)

				#Calculating the Chi squared values for the linear and quadratic fits.
				Chi2_l = np.sum(Chisqd_Lin(C_lin,logNu,logS,e_logS))
				Chi2_q = np.sum(Chisqd_Quad(C_quad,logNu,logS,e_logS))
		
				#Calculating the reduced Chir squared values for the linear and quadratic fits.
				Red_Chisqd_l = round(Chi2_l/(len(logNu)-len(C_lin)),3)
				Red_Chisqd_q = round(Chi2_q/(len(logNu)-len(C_quad)),3)
	
				#Calculating the 300Mhz flux with associated errors.
				l_S300 = 10**(Lin_mod(C_lin,np.log10(300*10e+5)))
				q_S300 = 10**(Quad_mod(C_quad,np.log10(300*10e+5)))
				e_l_S300 = l_S300*log_lin_e_S(C_e_lin)*np.log10(10)
				e_q_S300 = q_S300*log_quad_e_S(C_e_quad)*np.log10(10)
		
				if abs(Red_Chisqd_l - 1) >= abs(Red_Chisqd_q - 1):
					#Corresponds to the case where quad fit is better.
					flag = 3
				elif abs(Red_Chisqd_l - 1) < abs(Red_Chisqd_q - 1):
					#Corresponds to the case where linear fit is better.
					flag = 4
				
				if len(logNu) == 4:
					flag = 2#Pietro's sources
	
				#Assigning data values to the output tables.
				Name_List[i] = t_sample[i]['Name']
				val_list = [t_sample[i]['RA'],t_sample[i]['DEC'],l_S300,e_l_S300,q_S300,e_q_S300,\
Red_Chisqd_l,Red_Chisqd_q,C_lin[0],C_lin[1],C_e_lin[0],C_e_lin[1],C_quad[0],C_quad[1],C_quad[2],C_e_quad[0],\
C_e_quad[1],C_e_quad[2],flag]
	
				for v in range(len(val_list[2:])):#Formatting data
					val_list[2+v] = round(val_list[2+v],5)
				
				for l in range(len(val_list)):
					Out_List[i][l] = val_list[l]

		except (ValueError,NameError,IOError,KeyError,SyntaxError,TypeError) as e:
			err_file.write("\n \n {0} | {1} | Error: {2}, in fit.\n All values set to zero..\n \n \n".format(i,t_sample[i]['Name'],e))
			err_file.write(str(t_sample[i]))
			err_file.write("\n \n Table Values: \n %s" % str(val_list))
			err_file.write("\n Number of data points = %s" % len(logNu))

			flag = 5#Flag for non standard errors.
			Name_List[i] = t_sample[i]['Name']
			val_list = [t_sample[i]['RA'],t_sample[i]['DEC'],0,0,0,0,0,\
0,0,0,0,0,0,0,0,0,0,0,flag]

			#Here I am assigning the relevant values to the output array Out_List.
			for l in range(len(val_list)):
				Out_List[i][l] = val_list[l]	

		out_file.write("\n flag = {0}, lin reduced Chisqd = {1} \n quad reduced Chisqd = {2}\n".format(flag,Red_Chisqd_l,Red_Chisqd_q))
		#########################################################################################################################################

		#For this section a plot is generated for a random sample of the total catlogue.
		X = rand_sample_list - i
		#Formats the plotting so scales are relevant to the associated data.
		if len(logNu) == 0:
			log_Freq = np.linspace(np.log10(70*10e+5), np.log10(310*10e+5)+0.1, 1000)
		elif logNu[len(logNu)-1] < np.log10(300*10e+5) and len(logNu) > 0:
			log_Freq = np.linspace(logNu[0]-0.1, np.log10(310*10e+5)+0.1, 1000)
		elif logNu[len(logNu)-1] > np.log10(300*10e+5) and len(logNu) > 0:
			log_Freq = np.linspace(logNu[0]-0.1, logNu[len(logNu)-1]+0.1, 1000)
	
		if X[find(X,0)] == 0:
			plt.clf()
			if (flag == 3 or flag == 4) or flag == 2:
				plt.plot(10**log_Freq,10**Quad_mod(C_quad,log_Freq),label=r'$\rm{quad\:model}$')
				plt.plot(10**log_Freq,10**Lin_mod(C_lin,log_Freq),label=r'$\rm{lin\:model}$')
				plt.errorbar(300*10e+5,q_S300,yerr=e_q_S300,fmt='s',color="red",label=r'$\rm{quad\:S_{300MHz}}$')
				plt.title(r"$\rm{ %s ,\:\chi_{l,\nu}^2\:=\: %s ,\:\chi_{q,\nu}^2\:=\: %s,\:flag\:=\: %s }$" % (t_sample[i]["Name"],Red_Chisqd_l,Red_Chisqd_q,flag))
			elif flag == 1:
				plt.plot(10**log_Freq,10**Lin_mod(C_lin,log_Freq),label=r'$\rm{lin\:model}$')
				plt.title(r"$\rm{ %s ,\:\chi_{l,\nu}^2\:=\: %s ,\:flag\:=\: %s}$" % (t_sample[i]["Name"],Red_Chisqd_l,flag))

			plt.loglog()
			plt.errorbar(10**logNu,10**logS,yerr=logS*e_logS*np.log10(10),fmt='s',color='blue')
			plt.errorbar(300*10e+5,l_S300,yerr=e_l_S300,fmt='s',color="green",label=r'$\rm{lin\:S_{300MHz}}$')
			plt.xlabel(r"$\rm{\log_{10}(\nu)\:Hz}$",fontsize = 14)
			plt.ylabel(r"$\rm{\log_{10}(S_\nu)\:Jy}$",fontsize = 14)
			plt.legend()
			plt.savefig("./Fit_{0}/plots/{1}_{2}.png".format(sys.argv[1],i,t_sample['Name'][i]))	
#########################################################################################################################################
	#This next section takes the output arrays splits their colomns and assigns each column as an element
	#of a list using the function List_Split. This then returns a list with those column elements.
	#This list has the necessary formating to create a Table using astropy.Table alongside the Cols tuple.
	proto_Table = List_Split(Name_List,Out_List)
	Output_Table = Table(proto_Table,names=Cols,meta={'name':'first table'})
	Output_Table.write("./Fit_{0}/S300MHz_fit_{0}.fits".format(sys.argv[1]),"w")
	
	print "Table written to ./Fit_{0}".format(sys.argv[1])
Main()
