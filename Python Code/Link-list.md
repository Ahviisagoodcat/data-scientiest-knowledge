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
** Note: 342+465=807 **

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

445. Add Two Numbers 2

- Input: l1 = [2,4,3], l2 = [5,6,4]
- Output: [8,0,7] 

Hint: Reverse the link and add

### Reverse the Link

```
#self written:
 	def reverse_link(l):
        array1=[]
        while l:
            val=l.val
            l=l.next
            array1.append(val)
            
        result=n=ListNode(val)
        re_=array1[::-1]
        for i in range (1,len(re_)):
            n.next=ListNode(re_[i])
            n=n.next
        return result

#Other solution:
	def reverse_link(l):
		last=None #initial
		while l:
			tem=l.next #a temp break
			l.next=last #reverse
			last=l 
			l=tem #niddle
		return last

```

The other part:
```
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        #reverse list
        #add two numbers
        l1_r=self.reverse_list(l1)
        l2_r=self.reverse_list(l2)
        
        root=result=ListNode(0)
        res=0
        while l1_r or l2_r or res:
            v1=v2=0
            if l1_r:
                v1=l1_r.val
                l1_r=l1_r.next
            if l2_r:
                v2=l2_r.val
                l2_r=l2_r.next
            val=(v1+v2+res)%10
            res=(v1+v2+res)//10
            result.next=ListNode(val)
            result=result.next
        
        return self.reverse_list(root.next)
```
Other method: using the direct sum and convert to list
```
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        #not reverse
        s1=s2=0
        while l1:
            s1=s1*10+l1.val
            l1=l1.next
        while l2:
            s2=s2*10+l2.val
            l2=l2.next
            
        n=result=ListNode(0)
        for i in str(s1+s2):
            result.next=ListNode(i)
            result=result.next
        return n.next
```




