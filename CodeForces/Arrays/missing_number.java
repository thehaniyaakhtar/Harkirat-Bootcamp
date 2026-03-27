import java.util.*;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        
        int t = sc.nextInt();
        
        while(t-- > 0){
            int n = sc.nextInt();
            HashSet<Integer> set = new HashSet<>();
            
            for(int i = 0; i < n; i++){
                int x = sc.nextInt();
                
                if(set.contains(x)){
                    set.remove(x);
                }
                
                else{
                    set.add(x);
                }
            }
            
            for(int val : set){
                System.out.println(val);
            }
        }
    }
}
