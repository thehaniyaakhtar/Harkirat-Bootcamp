import java.util.*;
 
public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int n = sc.nextInt();
        int min = Integer.MAX_VALUE;
        int id = 0;
        
        for(int i = 0; i < n-1; i++){
            int t = sc.nextInt();
            
            if(t < min){
                min = t;
                id = i+1;
            }
            
            else if(t == min){
                id = i+1;
            }
        }
        System.out.println(id);
    }
}
