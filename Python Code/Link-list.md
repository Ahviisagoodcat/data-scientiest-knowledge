# Link-list

## List helper:
```
- root=result=ListNode(0)
- result.next=ListNode(val)
- result=result.next
```

## List Structure:
- self.next=next
- self.val=val

### Example problems:

2. Add Two Numbers

l1=[2,4,3], l2=[5,6,4], return [7,0,8] 
\* Note: 342+465=807 *\


Code:
```
# Python
def addtwonum(self,l1,l2):
	result=root=ListNode(0)
	res=0
	while l1 or l2 or res:
		v1=v2=0
		if l1:
			v1=l1.val
			l1=l1.next
		if l2:
			v2=l2.val
			l2=l2.next

		val=(v1+v2+res)%10
		res=(v1+v2+res)//10 
		result.next=ListNode(val)
		result=result.next
	return root.next
```

