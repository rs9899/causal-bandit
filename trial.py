"""
X - list of variables eg. [X_1, X_2, ...]
vals - dict of values eg. {X_1: 0, X_2: 1, ...}
A - intervention eg {X_3: 1}

Returns 
	P(X_1 = vals[X_1], X_2 = vals[X_2], ... | A)
"""
def parents(x):
	if x==1:
		return [2,3,4]
	elif x==2:
		return [4]
	elif x==3:
		return []
	elif x==4:
		return []
	else:
		return []

def pa_assign(x):
	if x==1:
		return [{2:0,3:0,4:0},{2:0,3:0,4:1},{2:0,3:1,4:0},{2:1,3:0,4:0},{2:0,3:1,4:1}, {2:1,3:0,4:1},{2:1,3:1,4:0},{2:1,3:1,4:1}]
	elif x==2:
		return [{4:0},{4:1}]
	elif x==3:
		return []
	elif x==4:
		return []
	else:
		return []	

def prob_given_parent(x, z):
	return (0.5, 0.5)

def P(X, vals, A):
	print("enterP" , X , vals, A)
	if len(X) == 0:
		return 1
	var = X[0]
	if var in A:
		if vals[var] == A[var]:
			return P(X[1:], vals, A)
		else:
			return 0.0
	else:
		pa_var = parents(var)
		if len(pa_var) == 0:
			return 0.5 * P(X[1:], vals, A) 

		new_var = set(pa_var).union(set(X[1:])) 
		# pa_assign returns a list of all {parent: value} assignments
		# so z is a dict containing parent:value items
		print("HERE")
		print(pa_assign(var) , var, vals)
		valid_assign = [z for z in pa_assign(var) if all([z[i]==v for i,v in vals.items() if i in z])]
		print(valid_assign)
		prob = 0.0


		for z in valid_assign:
			new_vals = z
			new_vals.update(vals)
			prob += (prob_given_parent(var, z)[vals[var]] * P(list(new_var), new_vals, A) )
		return prob

if __name__ == '__main__':
	z = P([2,3], {2:1,3:1}, {4:0})
	print(z)