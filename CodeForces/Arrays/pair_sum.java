import java.util.*;

public class Main{
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        int t = sc.nextInt();
        
        while(t-- > 0){
            int n = sc.nextInt();
            int[] arr = new int[n];
            
            for(int i = 0; i < n; i++){
                arr[i] = sc.nextInt();
            }
            
            int X = sc.nextInt();
            
            HashMap<Integer, Integer> map = new HashMap<>();
            int count = 0;
            
            for(int i = 0; i < n; i++){
                int needed = X - arr[i];
                
                if(map.containsKey(needed)){
                    count += map.get(needed);
                }
                
                map.put(arr[i], map.getOrDefault(arr[i], 0) + 1);
            }
            System.out.println(count);
        }
    }
}
