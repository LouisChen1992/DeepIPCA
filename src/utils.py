def deco_print(line, end='\n'):
	print('>==================> ' + line, end=end)

def sharpe(r):
	return r.mean() / r.std()