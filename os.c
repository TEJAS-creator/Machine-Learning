lab3
1. Write a shell script to find whether a given file is the directory or regular file.
#!/bin/bash

echo "Enter file name:"
read fname

if [ -d "$fname" ]
then
    echo "It is a Directory"
elif [ -f "$fname" ]
then
    echo "It is a Regular File"
else
    echo "File does not exist"
fi


2. Write a shell script to list all files (only file names) containing the input pattern (string) in the folder entered by the user.
#!/bin/bash

echo "Enter folder name:"
read folder

echo "Enter pattern:"
read pattern

grep -l "$pattern" $folder/*


3. Write a shell script to replace all files with .txt extension with .text in the current directory recursively.
#!/bin/bash

find . -type f -name "*.txt" | while read file
do
    mv "$file" "${file%.txt}.text"
done


4. Write a shell script to calculate the Gross Salary.

GS = Basics + TA + 10% of Basics (floating point).

#!/bin/bash

echo "Enter Basic Salary:"
read basic

echo "Enter TA:"
read ta

gs=$(echo "$basic + $ta + 0.10*$basic" | bc)

echo "Gross Salary = $gs"


5. Write a shell script to copy all files with extension entered by user to a new folder.
#!/bin/bash

echo "Enter file extension (ex: txt):"
read ext

echo "Enter folder name:"
read folder

mkdir -p $folder

for file in *.$ext
do
    cp "$file" "$folder"
done

echo "Files copied successfully"


6. Write a shell script to modify all occurrences of “ex:” with “Example:” only when it occurs at start of line or after period.
#!/bin/bash

for file in *
do
    if [ -f "$file" ]
    then
        sed -i 's/^ex:/Example:/g; s/\. ex:/\. Example:/g' "$file"
    fi
done


7. Write a shell script which deletes all even numbered lines in a text file.
#!/bin/bash

echo "Enter file name:"
read fname

sed '2~2d' "$fname"






lab4
Create Child Process + PID, PPID + wait()
#include<stdio.h>
#include<unistd.h>
#include<sys/wait.h>

int main()
{
    pid_t pid = fork();

    if(pid==0)
    {
        printf("Child Process\n");
        printf("PID: %d\n",getpid());
        printf("PPID: %d\n",getppid());
    }
    else
    {
        wait(NULL);
        printf("Parent Process\n");
        printf("PID: %d\n",getpid());
        printf("Child PID: %d\n",pid);
    }

    return 0;
}



Load Previous Program using exec()
#include<stdio.h>
#include<unistd.h>

int main()
{
    if(fork()==0)
    {
        printf("Child loading program...\n");
        execl("./a.out","a.out",NULL);
    }
    else
    {
        printf("Parent running\n");
    }
}


Zombie Process
#include<stdio.h>
#include<unistd.h>
#include<stdlib.h>

int main()
{
    int pid = fork();

    if(pid==0)
    {
        printf("Child exiting\n");
        exit(0);
    }
    else
    {
        printf("Parent sleeping\n");
        sleep(20);
    }
}


multithread sorting
#include<stdio.h>
#include<pthread.h>

int a[100],n;

void *bubble()
{
    for(int i=0;i<n-1;i++)
        for(int j=0;j<n-i-1;j++)
            if(a[j]>a[j+1])
            {
                int t=a[j];
                a[j]=a[j+1];
                a[j+1]=t;
            }

    printf("Bubble sort completed\n");
}

void *selection()
{
    for(int i=0;i<n-1;i++)
    {
        int min=i;

        for(int j=i+1;j<n;j++)
            if(a[j]<a[min])
                min=j;

        int t=a[i];
        a[i]=a[min];
        a[min]=t;
    }

    printf("Selection sort completed\n");
}

int main()
{
    pthread_t t1,t2;

    printf("Enter number of elements: ");
    scanf("%d",&n);

    printf("Enter elements:\n");

    for(int i=0;i<n;i++)
        scanf("%d",&a[i]);

    pthread_create(&t1,NULL,bubble,NULL);
    pthread_create(&t2,NULL,selection,NULL);

    pthread_join(t1,NULL);
    pthread_join(t2,NULL);

    printf("Sorted array:\n");

    for(int i=0;i<n;i++)
        printf("%d ",a[i]);
}




multithread fibonnaci
#include<stdio.h>
#include<pthread.h>

int n;
int fib[100];

void *fibonacci()
{
    fib[0]=0;
    fib[1]=1;

    for(int i=2;i<n;i++)
        fib[i]=fib[i-1]+fib[i-2];
}

int main()
{
    pthread_t tid;

    printf("Enter number of terms: ");
    scanf("%d",&n);

    pthread_create(&tid,NULL,fibonacci,NULL);

    pthread_join(tid,NULL);

    printf("Fibonacci Series:\n");

    for(int i=0;i<n;i++)
        printf("%d ",fib[i]);
}


lab 5

fcfs
#include<stdio.h>

int main()
{
    int at[4]={0,3,4,9};
    int bt[4]={60,30,40,10};
    int ct[4],wt[4],tat[4];

    int time=0;

    for(int i=0;i<4;i++)
    {
        if(time<at[i])
            time=at[i];

        time+=bt[i];
        ct[i]=time;

        tat[i]=ct[i]-at[i];
        wt[i]=tat[i]-bt[i];
    }

    printf("P\tAT\tBT\tCT\tTAT\tWT\n");

    for(int i=0;i<4;i++)
        printf("P%d\t%d\t%d\t%d\t%d\t%d\n",
        i+1,at[i],bt[i],ct[i],tat[i],wt[i]);
}


rr
#include<stdio.h>

int main()
{
    int bt[4]={60,30,40,10};
    int rt[4],ct[4];
    int q=10,time=0,remain=4;

    for(int i=0;i<4;i++)
        rt[i]=bt[i];

    while(remain>0)
    {
        for(int i=0;i<4;i++)
        {
            if(rt[i]>0)
            {
                if(rt[i]>q)
                {
                    time+=q;
                    rt[i]-=q;
                }
                else
                {
                    time+=rt[i];
                    ct[i]=time;
                    rt[i]=0;
                    remain--;
                }
            }
        }
    }

    for(int i=0;i<4;i++)
        printf("P%d Completion Time = %d\n",i+1,ct[i]);
}


priority
#include<stdio.h>

int main()
{
    int bt[4]={60,30,40,10};
    int pr[4]={3,2,1,4};
    int done[4]={0};
    int time=0;

    for(int c=0;c<4;c++)
    {
        int max=-1,pos=-1;

        for(int i=0;i<4;i++)
        {
            if(!done[i] && pr[i]>max)
            {
                max=pr[i];
                pos=i;
            }
        }

        time+=bt[pos];
        printf("P%d finished at %d\n",pos+1,time);
        done[pos]=1;
    }
}


srtf
#include<stdio.h>

int main()
{
    int bt[4]={60,30,40,10};
    int rt[4],complete=0,t=0,min,pos;

    for(int i=0;i<4;i++)
        rt[i]=bt[i];

    while(complete<4)
    {
        min=999;

        for(int i=0;i<4;i++)
        {
            if(rt[i]>0 && rt[i]<min)
            {
                min=rt[i];
                pos=i;
            }
        }

        rt[pos]--;
        t++;

        if(rt[pos]==0)
        {
            complete++;
            printf("P%d finished at %d\n",pos+1,t);
        }
    }
}


multilevel
#include<stdio.h>

int main()
{
    int bt[4]={60,30,40,10};
    int q1=10,q2=20;

    for(int i=0;i<4;i++)
    {
        if(bt[i]>q1)
        {
            bt[i]-=q1;
            printf("P%d moved to Queue2\n",i+1);
        }
        else
            printf("P%d finished in Queue1\n",i+1);
    }

    for(int i=0;i<4;i++)
    {
        if(bt[i]>0)
        {
            if(bt[i]>q2)
            {
                bt[i]-=q2;
                printf("P%d moved to Queue3\n",i+1);
            }
            else
                printf("P%d finished in Queue2\n",i+1);
        }
    }
}






lab6

pallindrome check

#include<stdio.h>
#include<string.h>
#include<sys/ipc.h>
#include<sys/msg.h>

struct msg
{
    long type;
    int num;
};

int main()
{
    int msgid;
    struct msg m;

    msgid = msgget(1234,0666|IPC_CREAT);

    printf("Enter number: ");
    scanf("%d",&m.num);

    m.type=1;

    msgsnd(msgid,&m,sizeof(m),0);

    printf("Number sent to Process B\n");
}


#include<stdio.h>
#include<sys/ipc.h>
#include<sys/msg.h>

struct msg
{
    long type;
    int num;
};

int palindrome(int n)
{
    int r,rev=0,temp=n;

    while(n>0)
    {
        r=n%10;
        rev=rev*10+r;
        n/=10;
    }

    return temp==rev;
}

int main()
{
    int msgid;
    struct msg m;

    msgid = msgget(1234,0666|IPC_CREAT);

    msgrcv(msgid,&m,sizeof(m),1,0);

    if(palindrome(m.num))
        printf("Palindrome\n");
    else
        printf("Not Palindrome\n");
}






producer using fifo
#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
#include<sys/stat.h>

int main()
{
    int fd,data[4];

    mkfifo("fifo",0666);

    fd=open("fifo",O_WRONLY);

    printf("Enter 4 numbers:\n");

    for(int i=0;i<4;i++)
        scanf("%d",&data[i]);

    write(fd,data,sizeof(data));

    close(fd);
}



consumer-
  #include<stdio.h>
#include<fcntl.h>
#include<unistd.h>

int main()
{
    int fd,data[4];

    fd=open("fifo",O_RDONLY);

    read(fd,data,sizeof(data));

    printf("Numbers received:\n");

    for(int i=0;i<4;i++)
        printf("%d ",data[i]);

    close(fd);
}




shared memory
#include<stdio.h>
#include<sys/ipc.h>
#include<sys/shm.h>
#include<unistd.h>

int main()
{
    int shmid;
    char *str;

    shmid = shmget(1234,100,0666|IPC_CREAT);

    if(fork()==0)
    {
        str = (char*)shmat(shmid,NULL,0);

        while(*str=='\0');

        *str = *str + 1;

        shmdt(str);
    }
    else
    {
        str = (char*)shmat(shmid,NULL,0);

        printf("Enter alphabet: ");
        scanf(" %c",str);

        sleep(2);

        printf("Next alphabet from child: %c\n",*str);

        shmdt(str);
        shmctl(shmid,IPC_RMID,NULL);
    }
}
