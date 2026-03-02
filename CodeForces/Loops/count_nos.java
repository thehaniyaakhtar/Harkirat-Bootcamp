import java.util.Scanner;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        
        int pos = 0;
        int neg = 0;
        int even = 0;
        int odd = 0;
        
        for(int i = 0; i < n; i++){
            int num = sc.nextInt();
            
            if(num > 0)
                pos++;
            
            if(num < 0)
                neg++;
            
            if(num%2==0)
                even++;
            
            if(num%2 != 0)
                odd++;
        }
        
        System.out.println(pos);
        System.out.println(neg);
        System.out.println(even);
        System.out.println(odd);
    }
}



