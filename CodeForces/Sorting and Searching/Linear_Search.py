import java.util.*;

public class Main{
    public static void main(String[] args){
        
        Scanner sc = new Scanner(System.in);
        
        int n = sc.nextInt();
        HashSet<Integer> set = new HashSet<>();
        
        for(int i = 0; i < n; i++){
            set.add(sc.nextInt());
        }
        
        int x = sc.nextInt();
        
        if(set.contains(x)) System.out.println("YES");
        else System.out.println("NO");
    }
}
